import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pickle
from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos2d, Reward, State
from ..utils.utils import merge01
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video, render_trajectory_with_radius
from .utils import get_lidar, inside_obstacles, get_node_goal_rng


class DubinsCar(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle
        timestep: jnp.ndarray # new field
        dist2tgt:jnp.ndarray
        min_dist2obs:Optional[jnp.ndarray] = jnp.array(0.0)
        edge_mask : jnp.ndarray = jnp.array(0)
        unsafe_mask:Optional[jnp.ndarray] = jnp.array(0).astype(jnp.bool_)
        safe_mask:Optional[jnp.ndarray] = jnp.array(0).astype(jnp.bool_)
        nearest_obs_vec:Optional[jnp.ndarray] = jnp.zeros((1, 2))  # (n_agents, 2): o_near - p, filled in by get_graph

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.4,       # R300 circumscribed radius (612x580mm body)
        "comm_radius": 1.5,
        "n_rays": 32,
        "obs_len_range": [0.4, 1.0],
        "n_obs": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.02,
            n_substeps: int = 5,
            params: dict = None,
            r_c_params: dict = None,
    ):
        super(DubinsCar, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True
        self.n_substeps = n_substeps

        self.action_low, self.action_high = self.action_lim()
        self.max_step = max_step
        self.reach_reward = r_c_params['reach_reward']
        self.w_delta1 = r_c_params['w_delta1']
        self.w_delta2 = r_c_params['w_delta2']
        #self.correction_cost_dist = r_c_params['correction_cost_dist']
        self.warning_dist2obs = r_c_params['warning_dist2obs']
        self.delta_action_scale = r_c_params['delta_action_scale']
        self.danger_penalty_coeff = r_c_params['danger_penalty_coeff']
        self.potential_obs_reward_coeff = r_c_params['potential_obs_reward_coeff']
        self.tgt_reward_coeff = r_c_params['tgt_reward_coeff']
        # --- Guidance (vortex field) reward parameters ---
        self.w_guide = r_c_params.get('w_guide', r_c_params['tgt_reward_coeff'])
        self.rho_min_coeff = r_c_params.get('rho_min_coeff', 1.2)
        self.lambda_guide = r_c_params.get('lambda_guide', 1.5)
        self.guide_reward_low = r_c_params.get('guide_reward_low', 0.0)
        self.guide_reward_high = r_c_params.get('guide_reward_high', 4.0)
        self.reward_scale = r_c_params['reward_scale']
        self.edge_coeff = r_c_params.get('edge_coeff', 2.0)
        self.cost_coeff = r_c_params['cost_coeff']
        self.cost = r_c_params['cost']
        self.cost_dist = r_c_params['cost_dist']
        self.cost_obs_dist = r_c_params['cost_obs_dist']

    @property
    def state_dim(self) -> int:
        return 4  # x, y, theta, v

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    def sample_non_overlapping_obstacles_numpy(self, key, n_obs, min_gap=0.05):
        """Do rejection sampling in numpy, then convert back to a jax array."""
        area = self.area_size
        lo, hi = self._params["obs_len_range"]

        # Define a pure-numpy inner function (does not receive traced values)
        def _sample_numpy(seed):
            # seed is a concrete int (when executed on the host)
            rng = np.random.default_rng(int(seed))
            centers, half_diags, ws, hs, thetas = [], [], [], [], []
            for _ in range(n_obs):
                for _ in range(500):
                    pos   = rng.uniform(0, area, size=2)
                    w     = rng.uniform(lo, hi)
                    h     = rng.uniform(lo, hi)
                    theta = rng.uniform(0, 2 * np.pi)
                    hd    = np.sqrt(w**2 + h**2) / 2
                    if len(centers) == 0:
                        break
                    dists    = np.linalg.norm(np.array(centers) - pos, axis=-1)
                    min_dist = np.array(half_diags) + hd + min_gap
                    if np.all(dists >= min_dist):
                        break
                centers.append(pos)
                half_diags.append(hd)
                ws.append(w); hs.append(h); thetas.append(theta)
            
            # Return numpy arrays (must have fixed shape and dtype)
            return (
                np.array(centers, dtype=np.float32),
                np.array(ws, dtype=np.float32),
                np.array(hs, dtype=np.float32),
                np.array(thetas, dtype=np.float32),
            )
        
        # Generate an int seed from the key
        numpy_seed = jax.random.randint(key, shape=(), minval=0, maxval=2147483647)

        # Declare the output shape and dtype (must be fixed!)
        result_shape_dtype = (
            jax.ShapeDtypeStruct((n_obs, 2), jnp.float32),  # centers
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # ws
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # hs
            jax.ShapeDtypeStruct((n_obs,), jnp.float32),    # thetas
        )
        
        # Use pure_callback to step outside the JAX trace
        all_pos, all_w, all_h, all_theta = jax.pure_callback(
            _sample_numpy,
            result_shape_dtype,
            numpy_seed,
            vmap_method='sequential',  # key: call one batch at a time under vmap
        )
        
        return self.create_obstacles(all_pos, all_w, all_h, all_theta)
    
    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        # randomly generate obstacles
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 2 * self.params["car_radius"], self.max_travel)

        # add random heading
        theta_key, key = jr.split(key, 2)
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
        states = states.at[:, 2].set(jr.uniform(theta_key, (self.num_agents,), minval=-np.pi, maxval=np.pi))
        goals = goals.at[:, 2].set(jnp.arctan2(goals[:, 1] - states[:, 1], goals[:, 0] - states[:, 0]))
        agent_pos = states[:, :2]
        goal_pos = goals[:, :2]
        dist2tgt = jnp.linalg.norm(agent_pos - goal_pos, axis=-1) 
        #min_dist2obs = self.get_min_dist_to_obstacles(agent_pos,obstacles)
        env_states = self.EnvState(states, goals, obstacles, jnp.array(0), jnp.array(dist2tgt), jnp.array(0.0), 
                                   jnp.array(0),jnp.array(0).astype(jnp.bool_),jnp.array(0).astype(jnp.bool_))
        graph, _ = self.get_graph(env_states)

        # save_var = {
        #     "obstacle": graph.env_states.obstacle,
        #     "initial_goal": graph.env_states.goal,
        #     "initial_agent": graph.env_states.agent,
        #     #"initial_graph": initial_graph
        # }
        # render_single_graph(graph, f"./initial_state_pkl/seed{123}_ep{13}.png", side_length=4.0,
        #                     n_agent=8, n_rays=32, r=0.05)
        # with open(f"./initial_state_pkl/seed{123}_ep{13}.pkl", "wb") as f:
        #     pickle.dump(save_var, f) 

        return graph

    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([
            (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (action[:, 0] * 1.0)[:, None],
            (action[:, 1])[:, None]
        ], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step(
            self, graph: EnvGraphsTuple, delta_action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1
        current_t = graph.env_states.timestep
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        pre_dist2tgt = graph.env_states.dist2tgt
        pre_min_dist2obs = graph.env_states.min_dist2obs
        u_ref = self.u_ref(graph)
        action = self.clip_action(self.delta_action_scale*delta_action+u_ref)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            stop_mask = 0 * stop_mask
        next_agent_states = jax.lax.fori_loop(
            0, self.n_substeps,
            lambda _, states: self.agent_step_euler(states, action, stop_mask),
            agent_states
        )
        next_t = current_t + 1
        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

         # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        #reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        
        # --- 4. Check termination conditions ---
        next_agent_pos = next_agent_states[:, :2]
        goal_pos = goal_states[:, :2]
        dist2tgt = jnp.linalg.norm(next_agent_pos - goal_pos, axis=-1) 
        is_reach = dist2tgt <  self._params["car_radius"] * 1 #0.15

        is_collision = inside_obstacles(next_agent_pos, obstacles, r=self._params["car_radius"])
        is_timeout = next_t >= self.max_step
        
        all_reach = jnp.all(is_reach)
        any_collision = jnp.any(is_collision)
        done = jnp.logical_or(any_collision, is_timeout)
        done = jnp.array(done)
        next_state = self.EnvState(next_agent_states, goal_states, obstacles, next_t, jnp.array(dist2tgt), jnp.array(0),jnp.array(0).astype(jnp.bool_),
                                jnp.array(0).astype(jnp.bool_))
        
        next_graph, extra_info = self.get_graph(next_state)
        cost = jnp.squeeze(self.get_cost(next_graph,any_collision))
        # --- 5. Compute the reward function ---
        reach_reward = jnp.where(all_reach, self.reach_reward, 0.0) # the reward can be smaller than the collision penalty
        assert cost.shape == tuple()
        assert done.shape == tuple()
        warning_dist = self._params["car_radius"] * self.warning_dist2obs# e.g. only activates within 4x the radius
        is_in_danger_zone = pre_min_dist2obs < warning_dist
        correction_cost = jnp.where(
            is_in_danger_zone,
            -self.w_delta2 * (jnp.linalg.norm(delta_action, axis=-1)**2),  # near an obstacle: correction is needed, so reduce the correction cost
            -self.w_delta1 * (jnp.linalg.norm(delta_action, axis=-1)**2)   # not near an obstacle: no correction needed, so penalize correction
        )

        # --- Guidance (vortex field) reward ---
        # Open area: d* points toward the goal (equivalent to tgt_reward without a lower-bound clip);
        # Danger zone: d* = goal direction + beta * lambda * tangential, turning "avoid the obstacle" into an explicit positive reward.
        eps = 1e-6
        start_pos = agent_states[:, :2]                                  # position at the start of this step (state observed by the policy)
        # Actual displacement this step (along the heading, since this car's velocity is naturally along the heading)
        disp = next_agent_pos - start_pos                                # (n_agents, 2)

        # The guidance field d* is computed entirely relative to "the start-of-step state"
        # Unit vector toward the goal (start -> goal)
        to_goal = goal_pos - start_pos                                   # (n_agents, 2)
        d_g = to_goal / (jnp.linalg.norm(to_goal, axis=-1, keepdims=True) + eps)

        # Unit vector toward the nearest obstacle (o_near - p), using the start-graph value
        obs_vec = graph.env_states.nearest_obs_vec                       # (n_agents, 2)
        n_hat = obs_vec / (jnp.linalg.norm(obs_vec, axis=-1, keepdims=True) + eps)

        # Tangential = rot90(n), take the side facing the goal
        t_raw = jnp.stack([-n_hat[:, 1], n_hat[:, 0]], axis=-1)          # rotate 90 degrees counterclockwise
        t_sign = jnp.sign(jnp.sum(t_raw * d_g, axis=-1, keepdims=True))
        t_sign = jnp.where(t_sign == 0.0, 1.0, t_sign)
        t_hat = t_raw * t_sign

        # Continuous beta blend: rho in [rho_min, rho_warn] maps linearly to [1, 0], using the start clearance
        rho = pre_min_dist2obs                                           # (n_agents,)
        rho_warn = self._params["car_radius"] * self.warning_dist2obs
        rho_min = self._params["car_radius"]*self.rho_min_coeff
        beta = jnp.clip((rho_warn - rho) / (rho_warn - rho_min + eps), 0.0, 1.0)  # (n_agents,)

        # Guidance direction d*
        d_star_raw = d_g + (self.lambda_guide * beta)[:, None] * t_hat
        d_star = d_star_raw / (jnp.linalg.norm(d_star_raw, axis=-1, keepdims=True) + eps)

        # Projection of the displacement onto the guidance direction; clip to bounds for stability
        guide_reward = self.w_guide * jnp.sum(disp * d_star, axis=-1)
        guide_reward = jnp.clip(guide_reward, self.guide_reward_low, self.guide_reward_high)
        # Combine the total reward
        reward_per_agent = (
            reach_reward +
            correction_cost +
            guide_reward
        )
        
        reward = reward_per_agent.mean()
        reward = reward/self.reward_scale
        reward = jnp.array(reward)
        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            info["inside_obstacles"] = is_collision
        info["dist2tgt"] = dist2tgt

        return next_graph, reward, cost, done, info


    # Assign cost based on distance
    def get_cost(self, graph: EnvGraphsTuple, any_collision) -> Cost:
        min_dist_to_obs = graph.env_states.min_dist2obs
        min_dist_to_obs = jnp.minimum(self._params["comm_radius"]*2.0, min_dist_to_obs)

        cost = jnp.maximum(0, self._params["car_radius"] * self.cost_obs_dist - min_dist_to_obs) * self.cost_coeff
        cost = jnp.where(min_dist_to_obs < self.cost_dist, self.cost, cost) # large cost on collision

        return cost

    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def render_trajectory(
        self, rollout: RolloutResult, save_path: pathlib.Path, dpi: int = 100, **kwargs
    ) -> None:
        render_trajectory(
            rollout=rollout,
            save_path=save_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            r=self.params["car_radius"],
            dt=self._dt,
            dpi=dpi,
            **kwargs
        )

    def render_trajectory_with_radius(
        self, rollout: RolloutResult, save_path: pathlib.Path, dpi: int = 100, **kwargs
    ) -> None:
        """Same as render_trajectory, but draws the car-radius circle along the trajectory (to show avoidance difficulty)."""
        render_trajectory_with_radius(
            rollout=rollout,
            save_path=save_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            r=self.params["car_radius"],
            dt=self._dt,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        pos_theta_diff = state.agent[:, None, :2] - state.agent[None, :, :2]
        agent_v = jnp.concatenate([(state.agent[:, 3] * jnp.cos(state.agent[:, 2]))[:, None],
                                   (state.agent[:, 3] * jnp.sin(state.agent[:, 2]))[:, None]], axis=-1)
        v_diff = agent_v[:, None, :] - agent_v[None, :, :]
        state_diff = jnp.concatenate([pos_theta_diff, v_diff], axis=-1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        agent_goal_pos_diff = state.agent[:, :2] - state.goal[:, :2]
        agent_goal_v_diff = agent_v
        agent_goal_edge_feats = jnp.concatenate([agent_goal_pos_diff, agent_goal_v_diff], axis=-1)
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_edge_feats = agent_goal_edge_feats.at[:, :2].set(agent_goal_edge_feats[:, :2] * coef)
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        for i in range(self.num_agents):
            agent_goal_edges.append(
                EdgeBlock(agent_goal_edge_feats[i][None, None, :], jnp.ones((1, 1)), id_agent[i][None], id_goal[i][None]))

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = jnp.concatenate([state.agent[i, :2], agent_v[i]]) - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"])
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([
            (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
            (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
            jnp.zeros((state.shape[0], 2))
        ], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.array([[1., 0.], [0., 1.]])], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], 4, 2)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        v = jnp.concatenate([(state[:, 3] * jnp.cos(state[:, 2]))[:, None],
                             (state[:, 3] * jnp.sin(state[:, 2]))[:, None]], axis=-1)
        edge_state = jnp.concatenate([state[:, :2], v], axis=-1)
        assert edge_state.shape[1] == self.edge_dim
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DubinsCar.GOAL)
        node_type = node_type.at[-n_hits:].set(DubinsCar.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 2))], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)
        # --- b. Compute the minimum distance from lidar_data ---
        agent_pos = state.agent[:, :2]
        agent_pos_expanded = jnp.expand_dims(agent_pos, axis=1)

        # all_lidar_data has shape (n_agents, n_rays, 2)
        # lidar_data is (n_agents * n_rays, 2) and needs reshaping
        all_lidar_data = lidar_data[:, :2].reshape(self.num_agents, self._params["n_rays"], 2)

        beam_vecs = all_lidar_data - agent_pos_expanded  # (n_agents, n_rays, 2): each ray hit point relative to the agent
        distances_all_beams = jnp.linalg.norm(beam_vecs, axis=-1)
        min_dists = jnp.min(distances_all_beams, axis=1)
        # Nearest-obstacle direction (o_near - p) for the guidance reward; missed rays are far and never enter the danger zone, so no special case is needed
        min_idx = jnp.argmin(distances_all_beams, axis=1)  # (n_agents,)
        nearest_obs_vec = beam_vecs[jnp.arange(self.num_agents), min_idx]  # (n_agents, 2)
        edge_mask = jnp.where(min_dists<self._params["car_radius"]*self.edge_coeff, 1, 0)
        state_with_dist = state._replace(min_dist2obs=min_dists, edge_mask=edge_mask, nearest_obs_vec=nearest_obs_vec)
        # create graph
        graph = GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state_with_dist,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()
    
        # --- c. Pack and return the extra info ---
        extra_info = {
            'min_dist_to_obs': min_dists
        }
        return graph, extra_info

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.5])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        lower_lim = jnp.ones(2) * -3.0
        upper_lim = jnp.ones(2) * 3.0
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        pos_diff = agent_states[:, :2] - goal_states[:, :2]

        # PID parameters
        k_omega = 1.0  # 0.5
        k_v = 2.3
        k_a = 2.5

        dist = jnp.linalg.norm(pos_diff, axis=-1)
        theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
        theta = agent_states[:, 2] % (2 * jnp.pi)
        theta_diff = theta_t - theta
        omega = jnp.zeros(agent_states.shape[0])
        agent_dir = jnp.concatenate([jnp.cos(theta)[:, None], jnp.sin(theta)[:, None]], axis=-1)
        assert agent_dir.shape == (agent_states.shape[0], 2)
        theta_between = jnp.arccos(
            jnp.clip(jnp.matmul(-pos_diff[:, None, :], agent_dir[:, :, None]).squeeze() / (dist + 0.0001),
                     a_min=-1, a_max=1))

        # when theta <= pi
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0), theta <= jnp.pi),
                          k_omega * theta_between, omega)
        # clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0)), theta <= jnp.pi),
            -k_omega * theta_between, omega
        )

        # when theta > pi
        # clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0), theta > jnp.pi),
                          -k_omega * theta_between, omega)
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0)), theta > jnp.pi),
            k_omega * theta_between, omega
        )

        omega = jnp.clip(omega, a_min=-5., a_max=5.)

        pos_diff_norm = jnp.sqrt(1e-6 + jnp.sum(pos_diff ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(pos_diff_norm, comm_radius)
        coef = jnp.where(pos_diff_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        pos_diff = coef * pos_diff
        a = -k_a * agent_states[:, 3] + k_v * jnp.linalg.norm(pos_diff, axis=-1)

        action = jnp.concatenate([omega[:, None], a[:, None]], axis=-1)
        action =self.clip_action(action)
        return action

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        heading_vec = jnp.concatenate([jnp.cos(agent_state[:, 2])[:, None],
                                       jnp.sin(agent_state[:, 2])[:, None]], axis=1)[:, None, :]
        heading_vec = heading_vec.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist ** 2 - 4 * self._params['car_radius'] ** 2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist ** 2 - self._params['car_radius'] ** 2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 0.5
        return stop
