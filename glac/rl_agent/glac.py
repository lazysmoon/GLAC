# glac.py
import gymnasium as gym
from .replay_buffer import PyTreeReplayBuffer
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import functools as ft
from glac.utils.typing import Array
from .utils import jax2np
from glac.custom_envs.base import MultiAgentEnv
# Import GNN-based networks
from glac.networks.policy_nets import ActorWithGNN, DoubleCriticWithGNN, LyapunovCritic
from .data import Rollout
from glac.custom_envs.base import RolloutResult
# Import graph data structure
from glac.utils.graph import GraphsTuple 
from glac.utils.utils import jax_jit_np, jax_vmap,tree_merge
import os
import time
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.cursive'] = ['Comic Sans MS']
matplotlib.rcParams['font.cursive'] = ['DejaVu Sans']  # Prevent fallback failures for 'cursive' font
# apply_tanh_correction function unchanged
def apply_tanh_correction(dist, action):
    log_prob = dist.log_prob(action).sum(axis=-1)
    log_prob -= (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))).sum(axis=-1)
    return log_prob


class GLACAgent:
    def __init__(self, 
                 # --- Environment and GNN-related parameters ---
                 env: MultiAgentEnv,
                 n_agents: int,
                 node_dim: int,
                 edge_dim: int,
                 state_dim: int,
                 action_dim: int,
                 # --- Base parameters ---
                 seed: int,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, hidden_dims=(256, 256),
                 use_lyapunov: bool = True,
                 lyapunov_loss_coeff: float =0.2,
                 alpha3: float = 0.8,):
        
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.use_lyapunov = use_lyapunov
        # Initialize PRNGKey
        self.key = jax.random.PRNGKey(seed)
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(self.key, 4)
        
        # --- 1. Create dummy graph for network initialization ---
        # Set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph
        
        # --- 2. Initialize ActorWithGNN ---
        actor_model = ActorWithGNN(action_dim=action_dim, 
                                   n_agents=self.n_agents, 
                                   hidden_dims=hidden_dims)
        
        actor_params = actor_model.init(actor_key, nominal_graph)['params']
        self.actor_state = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )
        
        # --- 3. Initialize DoubleCriticWithGNN ---
        dummy_actions = jnp.zeros((self.n_agents, action_dim)) # (n_agents, action_dim)
        
        critic_model = DoubleCriticWithGNN(n_agents=self.n_agents,
                                           hidden_dims=hidden_dims)
        
        critic_params = critic_model.init(critic_key, nominal_graph, dummy_actions)['params']
        self.critic_state = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.target_critic_params = critic_params
        
        # --- 4. Initialize Lyapunov critic ---
        if self.use_lyapunov:
            lyapunov_model = LyapunovCritic(n_agents=self.n_agents,hidden_dims=hidden_dims)

            #self.target_lyapunov_critic = LyapunovCritic(n_agents=self.n_agents,hidden_dims=hidden_dims)

            #self.target_lyapunov_critic.load_state_dict(self.lyapunov_critic.state_dict())
            lyapunov_params = lyapunov_model.init(lyapunov_key, nominal_graph, dummy_actions)['params']
            self.lyapunov_state = TrainState.create(
                apply_fn=lyapunov_model.apply,
                params=lyapunov_params,
                tx=optax.adam(learning_rate=critic_lr) # Reuse critic_lr
            )
            self.target_lyapunov_params = lyapunov_params

            # --- Initialize Lagrange multiplier Lambda ---
            self.log_lambda = jnp.array(0.0) # Initial log_lambda
            self.lambda_state = TrainState.create(
                apply_fn=None,
                params={'log_lambda': self.log_lambda},
                tx=optax.adam(learning_rate=actor_lr) # Reuse actor_lr
            )

            # self.log_niu = torch.tensor(np.log(1), requires_grad=True, device=self.device,dtype=torch.float32)
            # self.niu_optimizer = optim.Adam([self.log_niu], lr=alg_params['lr_a'])
        self.lyapunov_loss_coeff = lyapunov_loss_coeff
        self.alpha3 = alpha3
        # --- 5. Initialize Alpha (unchanged) ---
        self.target_entropy = -action_dim
        self.log_alpha = jnp.array(0.0) # Use scalar
        self.alpha_state = TrainState.create(
            apply_fn=None,
            params={'log_alpha': self.log_alpha},
            tx=optax.adam(learning_rate=alpha_lr)
        )
        
        # --- 6. JIT-compile update function ---
        # Note: batch is now a PyTree; JIT handles it accordingly
        self._update_step = jax.jit(self._update)
        #self._update_step = self._update

    @ft.partial(jax.jit, static_argnames=('self','deterministic'))
    def select_action(self, key, params, obs: GraphsTuple, deterministic: bool = False):
        # obs is a single graph (no batch dimension)
        dist = self.actor_state.apply_fn({'params': params}, obs)
        
        if deterministic:
            raw_action = dist.mean()
        else:
            raw_action = dist.sample(seed=key)
        
        # raw_action shape: (n_agents, action_dim)
        action = jnp.tanh(raw_action)
        
        # Remove batch and n_agents dimensions
        return action


    def update(self, main_batch: Rollout, edge_batch: Rollout): # batch is now a PyTree
        # _update_step is JIT-compiled; data is automatically transferred to device
        use_edge_batch_flag = 1.0 if edge_batch is not None else 0.0
        if edge_batch is None:
            edge_batch = jtu.tree_map(jnp.zeros_like, main_batch)
        key, self.key = jax.random.split(self.key, 2)
        self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params ,\
        self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, metrics = self._update_step(key,
                                self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params, 
                                self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, main_batch, edge_batch, jnp.array(use_edge_batch_flag))
        return metrics


    # _update function modified to handle GraphsTuple
    def _update(self, key, actor_state, critic_state, alpha_state, target_critic_params, 
                lyapunov_state, target_lyapunov_params, lambda_state, main_batch: Rollout, edge_batch: Rollout, use_edge_batch_flag):
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(key, 4)
        # Unpack graph data from batch
        obs = main_batch[0]      # Current observation graph
        actions = main_batch[1]   # Actions
        rewards = main_batch[2]   # Rewards
        costs = main_batch[3]     # Costs
        dones = main_batch[4]     # Done flags
        next_obs = main_batch[5]  # Next observation graph
        
        #if edge_batch:
        edge_obs = edge_batch[0]      # Current edge observation graph
        edge_actions = edge_batch[1]   # Edge actions
        edge_rewards = edge_batch[2]   # Edge rewards
        edge_costs = edge_batch[3]     # Edge costs
        edge_dones = edge_batch[4]     # Edge done flags
        edge_next_obs = edge_batch[5]  # Next edge observation graph
        
        def single_actor_forward(params, single_obs):
            # apply_fn expects a dict as the first argument
            return actor_state.apply_fn({'params': params}, single_obs)
        def single_critic_forward(params, single_obs, single_action):
            # apply_fn expects a dict as the first argument
            q1, q2 = critic_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(q1), jnp.squeeze(q2)
        next_dist_fn = jax.vmap(single_actor_forward, in_axes=(None, 0))
        next_q_fn = jax.vmap(single_critic_forward, in_axes=(None, 0, 0))
        # --- Critic update ---
        def critic_loss_fn(critic_params):
            
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=critic_key)
            
            # --- Apply Tanh correction ---
            next_log_probs = apply_tanh_correction(next_dist, next_raw_actions)
            next_actions = jnp.tanh(next_raw_actions)

            # ----------------------------------------
            next_q1, next_q2 = next_q_fn(target_critic_params, next_obs, next_actions)
            # Single-agent case: network outputs next_q1 of shape (5,1); squeeze to (5,)

            # next_q1, next_q2 = critic_state.apply_fn({'params': target_critic_params}, 
            #                                               next_obs, next_actions)
            
            next_q = jnp.minimum(next_q1, next_q2)
            alpha = jnp.exp(alpha_state.params['log_alpha'])
            # target_q shape: (batch_size, n_agents)
            next_log_probs = jnp.squeeze(next_log_probs)
            
            target_q = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_log_probs)
            
            # current_q1, current_q2 = critic_state.apply_fn({'params': critic_params}, 
            #                                                     obs, actions)
            current_q1, current_q2 = next_q_fn(critic_params, obs, actions)
            
            loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
            return loss, {'critic_loss': loss, 'q1': current_q1.mean(), 'q2': current_q2.mean()}
        
        (critic_loss_val, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)


        def single_lyapunov_forward(params, single_obs, single_action):
            # apply_fn expects a dict as the first argument
            lyapunov_value = lyapunov_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(lyapunov_value)
        
        next_lyapunov_fn = jax.vmap(single_lyapunov_forward, in_axes=(None, 0, 0))
        
        # --- Lyapunov Critic update ---
        # Use only data from the main replay buffer D
        def lyapunov_loss_fn(lyapunov_params):
            # Target: L' = c + gamma * L(s', a')
            # a' is the action taken by the current policy at s'
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=lyapunov_key)
            next_actions = jnp.tanh(next_raw_actions)
            
            # Use the target Lyapunov network
            l_next = next_lyapunov_fn(target_lyapunov_params, next_obs, next_actions)
            
            # cost 'c' is taken from the batch
            l_target = costs + self.gamma * (1 - dones) * l_next
            
            # Current L value
            l_current = next_lyapunov_fn(lyapunov_params, obs, actions)
            
            loss = ((l_current - l_target)**2).mean() * self.use_lyapunov
            return loss, {'lyapunov_loss': loss}

        (l_loss_val, l_metrics), l_grads = jax.value_and_grad(lyapunov_loss_fn, has_aux=True)(lyapunov_state.params)
        # Apply gradients conditionally based on flag
        # If use_lyapunov=False, gradients are zero and state remains unchanged
        new_lyapunov_state = lyapunov_state.apply_gradients(grads=l_grads)


        # --- Actor and Alpha update ---
        def actor_alpha_loss_fn(actor_params, alpha_params, lambda_params):


            dist_new = next_dist_fn(actor_params, obs)
            raw_actions_new = dist_new.sample(seed=actor_key)
            
            # --- Apply Tanh correction ---
            log_probs_new = apply_tanh_correction(dist_new, raw_actions_new)
            actions_new = jnp.tanh(raw_actions_new)

            q1, q2 = next_q_fn(new_critic_state.params, obs, actions_new)
            q = jnp.minimum(q1, q2)

            alpha_detached = jnp.exp(jax.lax.stop_gradient(alpha_params['log_alpha']))
            actor_loss_glac = (alpha_detached * jnp.squeeze(log_probs_new) - q).mean()


            alpha = jnp.exp(alpha_params['log_alpha'])
            log_probs_detached = jax.lax.stop_gradient(log_probs_new)
            alpha_loss = alpha * (-log_probs_detached.mean() - self.target_entropy)
            
            # b. Compute Lyapunov-related loss terms
            #if self.use_lyapunov:
            # Compute L(s, a) and L(s, a')
            # L(s,a) uses old actions from the buffer
            # actor_loss_lyapunov = 0
            # lambda_loss = 0
            # l_delta = 0
            edge_dist_next = next_dist_fn(actor_params, edge_next_obs)
            edge_raw_actions_next = edge_dist_next.sample(seed=actor_key)
            
            # --- Apply Tanh correction ---
            edge_log_probs_next = apply_tanh_correction(edge_dist_next, edge_raw_actions_next)
            edge_actions_next = jnp.tanh(edge_raw_actions_next)

            l_current_for_actor = jax.lax.stop_gradient(next_lyapunov_fn(new_lyapunov_state.params, edge_obs, edge_actions))
            # L(s,a') uses new actions generated by the Actor
            l_next_for_actor = next_lyapunov_fn(new_lyapunov_state.params, edge_next_obs, edge_actions_next)
            
            # l_delta = E[L(s, a') - L(s, a) + α₃*c(s,a)]
            # a = l_next_for_actor * jnp.squeeze(edge_next_obs.env_states.edge_mask)
            # b = (l_current_for_actor - self.alpha3 * edge_costs)*jnp.squeeze(edge_obs.env_states.edge_mask)
            #l_delta = (l_next_for_actor  - (l_current_for_actor - self.alpha3 * edge_costs) ).mean() * use_edge_batch_flag  # simplified l_delta
            l_delta = (l_next_for_actor * jnp.squeeze(edge_next_obs.env_states.edge_mask) -
                      (l_current_for_actor - self.alpha3 * edge_costs) * jnp.squeeze(edge_obs.env_states.edge_mask)).mean() * use_edge_batch_flag
            lambda_val  =  jnp.clip(jnp.exp(lambda_params['log_lambda']), 0, 20)
            actor_loss_lyapunov = (jax.lax.stop_gradient(lambda_val) * l_delta) * self.use_lyapunov * use_edge_batch_flag
            
            # d. Compute Lambda loss
            
            lambda_loss = - lambda_params['log_lambda'] * jax.lax.stop_gradient(l_delta) * self.use_lyapunov * use_edge_batch_flag
            
            actor_loss = actor_loss_glac + self.lyapunov_loss_coeff * actor_loss_lyapunov 
            total_loss = actor_loss + alpha_loss + lambda_loss
            return total_loss, (actor_loss, alpha_loss, lambda_loss, {'actor_loss': actor_loss, 
                                                         'alpha_loss': alpha_loss, 
                                                         'alpha': alpha, 
                                                         'entropy': -log_probs_detached.mean(),
                                                         'lambda_loss': lambda_loss, 
                                                         'lambda': lambda_val, 
                                                         'l_delta': l_delta})

        grad_fn = jax.value_and_grad(actor_alpha_loss_fn, argnums=(0, 1, 2), has_aux=True)
        ((_, (actor_loss_val, _, _, actor_alpha_metrics)), 
         (actor_grads, alpha_grads, lambda_grads)) = grad_fn(actor_state.params, alpha_state.params, lambda_state.params)
        
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_lambda_state = lambda_state.apply_gradients(grads=lambda_grads)
        
        # --- Soft update target Critic ---
        new_target_critic_params = jtu.tree_map(
            lambda target, online: target * (1 - self.tau) + online * self.tau,
            target_critic_params, new_critic_state.params
        )
   
        new_target_lyapunov_params = jtu.tree_map(
                lambda target, online: target * (1 - self.tau) + online * self.tau,
                target_lyapunov_params, new_lyapunov_state.params
            )
        metrics = {**critic_metrics, **l_metrics, **actor_alpha_metrics}
        
        return new_actor_state, new_critic_state, new_alpha_state, new_target_critic_params,\
               new_lyapunov_state, new_target_lyapunov_params, new_lambda_state, metrics
    
     # # --- Soft update target network (with Lyapunov) ---
        # def soft_update_fn():
        #     # If use_lyapunov is True, perform soft update
        #     return jtu.tree_map(
        #         lambda target, online: target * (1 - self.tau) + online * self.tau,
        #         target_lyapunov_params, new_lyapunov_state.params
        #     )

        # # def no_update_fn():
        # #     # If use_lyapunov is False, return old params directly
        # #     return target_lyapunov_params

        # # Use cond to select which branch to execute
        # new_target_lyapunov_params = jax.lax.cond(
        #     self.use_lyapunov,
        #     soft_update_fn,
        #     no_update_fn
        # )
    def save_agent_states(self, save_path, step, prefix="best_"):
        """Saves all TrainState objects."""
        
        save_data = {
            'actor': self.actor_state,
            'critic': self.critic_state,
            'alpha': self.alpha_state
        }
        # checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="best_")
        checkpoints.save_checkpoint(ckpt_dir=save_path, target=save_data, step=step, prefix=prefix)
        print(f"Agent states saved to directory: {save_path}")

    def load_agent_states(self, load_path):
        """Loads all TrainState objects from a directory."""
        
        # Create a template for restoring states
        template_states = {
            'actor': self.actor_state,
            'critic': self.critic_state,
            'alpha': self.alpha_state
        }
        
        loaded_states = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=template_states)
        print(f"Loading model from: {load_path}")
        self.actor_state = loaded_states['actor']
        self.critic_state = loaded_states['critic']
        self.alpha_state = loaded_states['alpha']
        self.target_critic_params = self.critic_state.params # Sync target network
        print(f"Agent states loaded from directory: {load_path}")


@ft.partial(jax.jit, static_argnames=('agent', 'eval_env', 'max_steps', 'seed', 'eval_episodes'))
def run_parallel_evaluation(
    agent, 
    eval_env, 
    max_steps: int, 
    seed: int, 
    actor_params,
    eval_episodes: int
):
    """
    Run multiple evaluation episodes in parallel efficiently using JAX vmap and scan.

    Args:
        agent: GLACAgent instance (static, provides methods)
        eval_env: JAX-compatible environment instance (static)
        max_steps: Maximum steps per episode (static)
        actor_params: Trained actor network parameters (dynamic)
        seed: Initial random seed
        eval_episodes: Number of episodes to run in parallel

    Returns:
        A JAX array containing the total reward for each episode.
    """

    # --- 1. Define core rollout logic for a single episode ---
    def rollout_single_episode(key):
        
        # a. Reset environment
        reset_key, rollout_key = jr.split(key)
        initial_graph = eval_env.reset(reset_key)
        
        # b. Define single-step scan body
        def step_fn(carry, _):
            # carry holds (current graph, cumulative reward, done flag)
            prev_graph, cumulative_reward, key, done_flag = carry
            
            # --- Use lax.cond to simulate early termination ---
            def do_step():
                # If not done, execute one normal step
                a_key, next_key  = jax.random.split(key)
                action = agent.select_action(a_key, actor_params, prev_graph, deterministic=True)
                next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
                transition = (prev_graph,action,reward,cost,done, next_graph)
                new_cumulative_reward = cumulative_reward + reward
                return next_graph, new_cumulative_reward, next_key, done, transition

            def skip_step():
                # If done, state and reward remain unchanged
                action = agent.select_action(key, actor_params, prev_graph, deterministic=True)
                transition = (prev_graph,action, jnp.array(0.0), jnp.array(0.0), done_flag, prev_graph)
                return prev_graph, cumulative_reward, key, done_flag, transition

            # Choose branch based on previous done_flag
            next_graph, new_cumulative_reward, new_key, current_done_signal, current_transition = jax.lax.cond(
                done_flag,
                skip_step,
                do_step
            )
            
            # Update done_flag: once True, stays True
            new_done_flag = jnp.logical_or(done_flag, current_done_signal)
            
            # Return carry for the next iteration
            return (next_graph, new_cumulative_reward, new_key, new_done_flag), current_transition

        # c. Set initial carry for scan
        initial_carry = (initial_graph, 0.0, rollout_key, jnp.array(False))
        
        # d. Run the entire episode using lax.scan
        final_carry, all_transition = jax.lax.scan(
            step_fn, 
            initial_carry, 
            None, 
            length=max_steps
        )
        
        # e. Extract final total reward
        final_graph, final_reward, _, _ = final_carry
        successful_flag = jnp.where(final_graph.env_states.dist2tgt<0.1,1,0)
        safe_flag = jnp.where(final_graph.env_states.timestep>=254,1,0)
        return successful_flag, safe_flag, final_reward
        
    # --- 2. Parallelize the rollout using vmap ---
    # a. Create an independent key for each parallel episode
    keys = jr.split(jax.random.PRNGKey(seed), eval_episodes)
    
    # b. vmap applies rollout_single_episode in parallel over all keys
    all_successful_flag, all_safe_flag, all_rewards = jax.vmap(rollout_single_episode)(keys)
    
    return all_successful_flag, all_safe_flag, all_rewards

class GLACTrainer:

    def __init__(
            self,
            env: MultiAgentEnv, 
            env_test: MultiAgentEnv,
            agent: GLACAgent,
            log_dir: str,
            seed: int,
            params: dict,
            configs: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.agent = agent
        graph = env.reset(jax.random.PRNGKey(0))
        #dummy_actions = np.array([[1,2]])
        dummy_actions = jnp.ones((env.num_agents, env.action_dim)) # (n_agents, action_dim)
        next_graph, reward, cost, done, info = env.step(graph, dummy_actions)
        dummy_transition = (graph,dummy_actions,reward,cost,done, next_graph)
        self.PyTreereplay_buffer = PyTreeReplayBuffer(capacity=int(1e6), dummy_input=dummy_transition)
        self.log_dir = os.path.abspath(log_dir)  # Ensure absolute path
        self.seed = seed
        self.action_low, self.action_high = env.action_lim()
        # Static parameter validation
        if GLACTrainer._check_params(params):
            self.params = params

        # Extract configuration from params
        self.total_steps = params['total_timesteps']
        self.start_steps = params['start_timesteps']
        self.batch_size = params['batch_size']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']
        self.train_per_cycle = params['train_per_cycle']
        
        self.save_log = save_log
        self.max_episode_steps = env.max_step
        self.horizon = 32
        # Create model checkpoint directory
        if self.save_log:
            self.model_dir = os.path.join(self.log_dir, 'models')
            os.makedirs(self.model_dir, exist_ok=True)
        
        wandb.login()
        wandb.init(
            name=params['run_name'], 
            project='GLAC_JAX_DoubleIntergtor', 
            dir=self.log_dir,
            config=configs
        )
        self.key = jax.random.PRNGKey(seed)
        world_model_key, self.key = jax.random.split(self.key)
        #self.world_model = WorldModel(params['world_model_cfg'], world_model_key)
        #self.world_model = GNNWorldModel(self.env, self.agent, params['world_model_cfg'], world_model_key)
        self.env_model_error = 0
        self.model_steps = 100
        self.update_steps = 0
        self.best_eval_reward = -np.inf
        self.best_successful_rate = 0

    @staticmethod
    def _check_params(params: dict) -> bool:
        # Validate required training parameters
        assert 'run_name' in params
        assert 'total_timesteps' in params
        assert 'start_timesteps' in params
        assert 'batch_size' in params
        assert 'eval_interval' in params and params['eval_interval'] > 0
        assert 'eval_epi' in params and params['eval_epi'] >= 1
        assert 'save_interval' in params and params['save_interval'] > 0
        return True
    
    @ft.partial(jax.jit, static_argnames=('self', 'agent', 'env','max_steps'))
    def rollout_single_episode(
        self,
        agent, 
        env, 
        max_steps: int,
        actor_params,
        key):
        
        # a. Reset environment
        reset_key, rollout_key = jr.split(key)
        initial_graph = env.reset(reset_key)
        initial_N = -1
        # b. Define single-step scan body
        def step_fn(carry, _):
            # carry holds (current graph, cumulative reward, done flag)
            prev_graph, cumulative_reward, cumulative_cost, key, done_flag, prev_N = carry
            
            # --- Use lax.cond to simulate early termination ---
            def do_step():
                # If not done, execute one normal step
                a_next_key, next_key = jax.random.split(key)
                action = agent.select_action(a_next_key, actor_params, prev_graph, deterministic=False)
                next_graph, reward, cost, done, info = env.step(prev_graph, action)
                current_N =  jnp.where(
                next_graph.env_states.min_dist2obs < 0.1,
                next_graph.env_states.timestep,
                prev_N #
                )
                transition = (prev_graph,action,reward,cost,done, next_graph)
                new_cumulative_reward = cumulative_reward + reward
                new_cumulative_cost = cumulative_cost + cost
                return next_graph, action, new_cumulative_reward, new_cumulative_cost, next_key, done, transition, jnp.squeeze(current_N), info

            def skip_step():
                # If done, state and reward remain unchanged
                action = jnp.zeros((env.num_agents, env.action_dim))
                next_graph, reward, cost, done, info = env.step(prev_graph, action)
                transition = (prev_graph,action, jnp.array(0.0), jnp.array(0.0), done_flag, prev_graph)
                return prev_graph, action, cumulative_reward, cumulative_cost, key, done_flag, transition, prev_N, info

            # Choose branch based on previous done_flag
            next_graph, action, new_cumulative_reward, new_cumulative_cost, new_key, current_done_signal, current_transition, current_N, info = jax.lax.cond(
                done_flag,
                skip_step,
                do_step
            )
            
            # Update done_flag: once True, stays True
            new_done_flag = current_done_signal
            
            # Return carry for the next iteration
            return (next_graph, new_cumulative_reward, new_cumulative_cost, new_key, new_done_flag, current_N), (current_transition, info)

        # c. Set initial carry for scan
        initial_carry = (initial_graph, 0.0, 0.0, rollout_key, jnp.array(False), initial_N)
        
        # d. Run the entire episode using lax.scan
        final_carry, (all_transition, infos) = jax.lax.scan(
            step_fn, 
            initial_carry, 
            None, 
            length=max_steps
        )
        
        # e. Extract final total reward
        _, final_reward, final_cost, key, _, edge_N= final_carry

        return final_reward, final_cost, all_transition, infos, edge_N
    

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> jnp.ndarray:
        # safe if in the horizon, the agent is always safe
        def safe_rollout( single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask).astype(jnp.bool_)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_mask = ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_)
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(safe_mask)
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(jnp.array(1).astype(jnp.bool_))
                #graph = graph.env_states._replace(safe_mask=safe_mask)
            
            return safe_rollout_mask

        safe = safe_rollout(graph.env_states.unsafe_mask)
        state_with_safe = graph.env_states._replace(safe_mask=safe)
        graph = graph._replace(env_states=state_with_safe)
        return graph
    def train(self):
        start_time = time.time()
        key_x0, self.key = jax.random.split(self.key)
        episode_num = 0
        current_step = 1
        steps_to_collect = self.env._max_step
        # is_unsafes = []
        # all_rollouts_for_video = []
        # all_rollouts_for_videos = None
        # model_dir = './logs/DoubleIntegrator/gcbf+/seed0_202509081722/models'
        # videos_dir = os.path.join(model_dir, f"vd_train")
        # os.makedirs(videos_dir, exist_ok=True)
        # is_unsafe_fn = jax_jit_np(jax_vmap(self.env.collision_mask))
        collect_time = 0
        pbar = tqdm(total=int(self.total_steps), ncols=80)

        while current_step < self.total_steps + 1:
            
            key_x0, self.key = jax.random.split(self.key)
            # Update world model (currently disabled)
            # if self.PyTreereplay_buffer.size > 3000:
            #     losses = self.world_model.fit(self.PyTreereplay_buffer, steps=self.model_steps)
            #     self.env_model_error = np.mean(losses)
            #     wandb.log({
            #     "train/world_loss": self.env_model_error,
            #     }, step = self.world_model.update_steps)
            episodes_return, episodes_cost, all_transitions, summaries, edge_N = self.rollout_single_episode(self.agent, self.env, 
                                                    self.env.max_step,self.agent.actor_state.params, key_x0)
            
            (graph,action,reward,cost,done, next_graph) = all_transitions
            done_indices = np.where(done)[0]
            done_index = done_indices[0]
            episode_reward = episodes_return
            episode_length = done_index+1

            infos_np = jtu.tree_map(np.asarray, summaries)
            dist2tgt = infos_np['dist2tgt'][done_index]

            # T_... data has length T
            episode_transitions = jtu.tree_map(
                lambda x: x[0:done_index+1],
                all_transitions
            )
            
            # Convert JAX arrays to NumPy
            summaries_np = jtu.tree_map(np.asarray, summaries)         
            dist2tgt = summaries_np['dist2tgt'][done_index]
            # Log episode information
            episode_verbose = ( f"Episode finished at step {current_step + done_index}: "
                                f"Episode_Reward={episode_reward:.2f}, "
                                f"Episode_Cost={episodes_cost:.2f}, "
                                f"Episode_Length={episode_length}, "
                                f"dist2tgt={dist2tgt}")
            tqdm.write(episode_verbose)
            wandb.log({
                "rollout/episode_reward": episode_reward,
                "rollout/episode_length": episode_length
            }, step=current_step + done_index)
            episode_num += 1

            (graph,action,reward,cost,done, next_graph) = episode_transitions
            graph_with_unsafemask = (jax_vmap(self.env.unsafe_mask))(episode_transitions[0])
            graph_with_safemask = self.safe_mask(graph_with_unsafemask)
            episode_transitions = (graph_with_safemask,action,reward,cost,done, next_graph)

            
            self.PyTreereplay_buffer.add_batch(episode_transitions)
            self.PyTreereplay_buffer.add_edge(edge_N, episode_transitions)
            current_step += episode_length
            # --- 2. Algorithm update ---
            if current_step >= self.start_steps and self.PyTreereplay_buffer.size > self.batch_size:
                train_per_cycle = 80
                for _ in range(train_per_cycle):
                    main_batch, edge_batch = self.PyTreereplay_buffer.sample(self.batch_size)
                    update_info = self.agent.update(main_batch, edge_batch)
                    # Log training metrics
                    if self.update_steps % 100 == 0: # Reduce logging frequency
                        wandb.log({f"train/{k}": v.item() for k, v in update_info.items()}, step=current_step)
                    self.update_steps += 1
            
            # --- 3. Evaluation and model saving ---
            if collect_time % self.eval_interval == 0:
                # Run parallel evaluation
                all_successful_flag, all_safe_flag, all_episode_rewards = run_parallel_evaluation( agent=self.agent,
                eval_env=self.env_test,
                max_steps=self.max_episode_steps,
                eval_episodes=self.eval_epi,
                actor_params=self.agent.actor_state.params,  # Pass the latest params
                seed=self.seed )
                all_episode_rewards_np = np.array(all_episode_rewards)
                # Compute averages
                eval_successful_rate = all_successful_flag.mean().item()
                eval_safe_rate = all_safe_flag.mean().item()
                eval_reward = all_episode_rewards_np.mean()
                wandb.log({"eval/mean_reward": eval_reward}, step=current_step)
                wandb.log({"eval/eval_successful_rate": eval_successful_rate}, step=current_step)
                wandb.log({"eval/eval_safe_rate": eval_safe_rate}, step=current_step)
                
                time_since_start = time.time() - start_time
                eval_verbose = (f'Step: {current_step}, Time: {time_since_start:.0f}s, Eval Reward: {eval_reward:.2f}')
                tqdm.write(eval_verbose)

                if self.save_log:
                    # Save best model
                    if eval_reward > self.best_eval_reward :#and eval_successful_rate > self.best_successful_rate
                        self.best_eval_reward = eval_reward
                        self.best_successful_rate = eval_successful_rate
                        success_text = f"Success Rate : {eval_successful_rate*100} %\n"
                        eval_dir = os.path.join(self.model_dir, f"eval_train_best")
                        os.makedirs(eval_dir, exist_ok=True)
                        # Write to file (create if not exists)
                        txt_path = os.path.join(eval_dir, f"output.txt")
                        with open(txt_path, "w", encoding="utf-8") as file:
                            file.writelines([success_text])
                        tqdm.write(f"New best model found! Saving...")
                        #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="best_")
                        self.agent.save_agent_states(self.model_dir, current_step, prefix="best_")
                    
            # Periodic checkpoint save
            if collect_time % self.save_interval == 0 and self.save_log:
                tqdm.write(f"Saving interval checkpoint...")
                self.agent.save_agent_states(self.model_dir, current_step, prefix="checkpoint_")
                #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="checkpoint_")
            collect_time += 1
            pbar.update(episode_length)

        # Save final model
        if self.save_log:
            print("Training finished. Saving final model.")
            self.agent.save_agent_states(self.model_dir, current_step, prefix="final_")
            #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), self.total_steps, prefix="final_")

        wandb.finish()