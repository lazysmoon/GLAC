# glac.py
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
# GNN-based networks
from glac.networks.policy_nets import ActorWithGNN, DoubleCriticWithGNN, LyapunovCritic
from .data import Rollout
from glac.custom_envs.base import RolloutResult
# Graph data structure
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
matplotlib.rcParams['font.cursive'] = ['DejaVu Sans']  # key line: avoid failing to find 'cursive'
def apply_tanh_correction(dist, action):
    log_prob = dist.log_prob(action).sum(axis=-1)
    log_prob -= (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))).sum(axis=-1)
    return log_prob


class GLACAgent:
    def __init__(self, 
                 # --- New environment and GNN related parameters ---
                 env: MultiAgentEnv,
                 n_agents: int,
                 node_dim: int,
                 edge_dim: int,
                 state_dim: int,
                 action_dim: int,
                 # --- Original parameters ---
                 seed: int,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, hidden_dims=(256, 256),
                 use_lyapunov: bool = True,
                 lyapunov_loss_coeff: float =0.2,
                 alpha3: float = 0.8,
                 # --- Stabilization-related parameters ---
                 lambda_lr: float = 1e-4,        # separate (smaller) lr for lambda, suppresses windup
                 lambda_clip_max: float = 10.0,  # upper bound on lambda, prevents the multiplier from overwhelming the actor
                 l_delta_margin: float = 0.05,   # constraint margin epsilon: only penalize the part where l_delta > epsilon, removing the root cause of windup
                 huber_delta: float = 10.0,      # delta of the critic Huber loss, degrades to linear when the TD error explodes
                 q_clip_min: float = -100.0,     # lower clip bound for the critic target_q
                 q_clip_max: float = 100.0,      # upper clip bound for the critic target_q
                 max_grad_norm: float = 10.0,):  # global gradient norm clip bound (<=0 disables it)

        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.use_lyapunov = use_lyapunov
        self.lambda_clip_max = lambda_clip_max
        self.l_delta_margin = l_delta_margin
        self.huber_delta = huber_delta
        self.q_clip_min = q_clip_min
        self.q_clip_max = q_clip_max

        # Prepend clip_by_global_norm before adam: clip the global norm of the whole
        # gradient tree to max_grad_norm, preventing an abnormally large single-step
        # gradient from blowing up the network (especially needed for the critic).
        def make_optimizer(lr):
            if max_grad_norm and max_grad_norm > 0:
                return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
            return optax.adam(lr)
        self._make_optimizer = make_optimizer
        # Initialize the PRNGKey
        self.key = jax.random.PRNGKey(seed)
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(self.key, 4)

        # --- 1. Create a dummy graph for initialization ---
        # set nominal graph for initialization of the neural networks
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
            tx=self._make_optimizer(actor_lr)
        )
        
        # --- 3. Initialize DoubleCriticWithGNN ---
        dummy_actions = jnp.zeros((self.n_agents, action_dim)) # (n_agents, action_dim)
        
        critic_model = DoubleCriticWithGNN(n_agents=self.n_agents,
                                           hidden_dims=hidden_dims)
        
        critic_params = critic_model.init(critic_key, nominal_graph, dummy_actions)['params']
        self.critic_state = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=self._make_optimizer(critic_lr)
        )
        self.target_critic_params = critic_params
        
        # -- 4. Initialize the Lyapunov critic
        if self.use_lyapunov:
            lyapunov_model = LyapunovCritic(n_agents=self.n_agents,hidden_dims=hidden_dims)

            #self.target_lyapunov_critic = LyapunovCritic(n_agents=self.n_agents,hidden_dims=hidden_dims)

            #self.target_lyapunov_critic.load_state_dict(self.lyapunov_critic.state_dict())
            lyapunov_params = lyapunov_model.init(lyapunov_key, nominal_graph, dummy_actions)['params']
            self.lyapunov_state = TrainState.create(
                apply_fn=lyapunov_model.apply,
                params=lyapunov_params,
                tx=self._make_optimizer(critic_lr) # critic_lr can be reused
            )
            self.target_lyapunov_params = lyapunov_params

            # --- Initialize the Lagrange multiplier Lambda ---
            self.log_lambda = jnp.array(0.0) # initial log_lambda
            self.lambda_state = TrainState.create(
                apply_fn=None,
                params={'log_lambda': self.log_lambda},
                tx=optax.adam(learning_rate=lambda_lr) # use the smaller lambda_lr to suppress windup
            )

            # self.log_niu = torch.tensor(np.log(1), requires_grad=True, device=self.device,dtype=torch.float32)
            # self.niu_optimizer = optim.Adam([self.log_niu], lr=alg_params['lr_a'])
        self.lyapunov_loss_coeff = lyapunov_loss_coeff
        self.alpha3 = alpha3
        # --- 5. Initialize Alpha ---
        self.target_entropy = -action_dim
        self.log_alpha = jnp.array(0.0) # use a scalar
        self.alpha_state = TrainState.create(
            apply_fn=None,
            params={'log_alpha': self.log_alpha},
            tx=optax.adam(learning_rate=alpha_lr)
        )
        
        # --- 6. JIT compile the update function ---
        # Note: batch is now a Pytree, so JIT must be told about it
        self._update_step = jax.jit(self._update)
        #self._update_step = self._update

    @ft.partial(jax.jit, static_argnames=('self','deterministic'))
    def select_action(self, key, params, obs: GraphsTuple, deterministic: bool = False):
        # The input obs is already a single graph (no batch dimension)
        dist = self.actor_state.apply_fn({'params': params}, obs)
        
        if deterministic:
            raw_action = dist.mean()
        else:
            raw_action = dist.sample(seed=key)
        
        # raw_action_batch shape: (n_agents, action_dim)
        action = jnp.tanh(raw_action)

        return action


    def update(self, main_batch: Rollout, edge_batch: Rollout): # batch is now a Pytree
        # _update_step is JIT compiled, so data is transferred to the device automatically
        use_edge_batch_flag = 1.0 if edge_batch is not None else 0.0
        if edge_batch is None:
            edge_batch = jtu.tree_map(jnp.zeros_like, main_batch)
        key, self.key = jax.random.split(self.key, 2)
        self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params ,\
        self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, metrics = self._update_step(key,
                                self.actor_state, self.critic_state, self.alpha_state, self.target_critic_params, 
                                self.lyapunov_state, self.target_lyapunov_params, self.lambda_state, main_batch, edge_batch, jnp.array(use_edge_batch_flag))
        return metrics


    # _update also needs to be adapted to handle GraphsTuple
    def _update(self, key, actor_state, critic_state, alpha_state, target_critic_params,
                lyapunov_state, target_lyapunov_params, lambda_state, main_batch: Rollout, edge_batch: Rollout, use_edge_batch_flag):
        actor_key, critic_key, lyapunov_key, alpha_key = jax.random.split(key, 4)
        # Unpack the graph data from the batch
        obs = main_batch[0]      # current observation graph
        actions = main_batch[1]  # actions
        rewards = main_batch[2]  # rewards
        costs = main_batch[3]    # costs
        dones = main_batch[4]    # done flags
        next_obs = main_batch[5] # next observation graph

        edge_obs = edge_batch[0]      # current observation graph
        edge_actions = edge_batch[1]  # actions
        edge_rewards = edge_batch[2]  # rewards
        edge_costs = edge_batch[3]    # costs
        edge_dones = edge_batch[4]    # done flags
        edge_next_obs = edge_batch[5] # next observation graph

        def single_actor_forward(params, single_obs):
            # apply_fn expects a dict as its first argument
            return actor_state.apply_fn({'params': params}, single_obs)
        def single_critic_forward(params, single_obs, single_action):
            # apply_fn expects a dict as its first argument
            q1, q2 = critic_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(q1), jnp.squeeze(q2)
        next_dist_fn = jax.vmap(single_actor_forward, in_axes=(None, 0))
        next_q_fn = jax.vmap(single_critic_forward, in_axes=(None, 0, 0))
        # --- Critic update ---
        def critic_loss_fn(critic_params):
            
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=critic_key)
            
            # --- Tanh correction applied to the critic target computation ---
            next_log_probs = apply_tanh_correction(next_dist, next_raw_actions)
            next_actions = jnp.tanh(next_raw_actions)

            # ----------------------------------------
            next_q1, next_q2 = next_q_fn(target_critic_params, next_obs, next_actions)
            # Single-agent case: the network outputs next_q1 of shape (5, 1); squeeze it to (5,)

            next_q = jnp.minimum(next_q1, next_q2)
            alpha = jnp.exp(alpha_state.params['log_alpha'])
            # target_q shape: (batch_size, n_agents)
            next_log_probs = jnp.squeeze(next_log_probs)

            target_q = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_log_probs)
            # The target is independent of critic_params; after explicit stop_gradient, clip it to bounds to truncate exploding TD targets
            target_q = jax.lax.stop_gradient(jnp.clip(target_q, self.q_clip_min, self.q_clip_max))

            current_q1, current_q2 = next_q_fn(critic_params, obs, actions)

            # Huber/smooth-L1: degrades to linear when the TD error is large, avoiding the 1e6-level positive feedback caused by MSE squaring
            loss = (optax.huber_loss(current_q1, target_q, delta=self.huber_delta) +
                    optax.huber_loss(current_q2, target_q, delta=self.huber_delta)).mean()
            return loss, {'critic_loss': loss, 'q1': current_q1.mean(), 'q2': current_q2.mean()}
        
        (critic_loss_val, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_metrics['critic_grad_norm'] = optax.global_norm(critic_grads)  # raw gradient norm before clipping
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)


        def single_lyapunov_forward(params, single_obs, single_action):
            # apply_fn expects a dict as its first argument
            lyapunov_value = lyapunov_state.apply_fn({'params': params}, single_obs, single_action)
            return jnp.squeeze(lyapunov_value)

        next_lyapunov_fn = jax.vmap(single_lyapunov_forward, in_axes=(None, 0, 0))

        # --- Lyapunov critic update ---
        # Also uses only data from the main buffer D
        def lyapunov_loss_fn(lyapunov_params):
            # Target L' = c + gamma * L(s', a')
            # a' is the current policy's action at s'
            next_dist = next_dist_fn(actor_state.params, next_obs)
            next_raw_actions = next_dist.sample(seed=lyapunov_key)
            next_actions = jnp.tanh(next_raw_actions)

            # Use the target Lyapunov network
            l_next = next_lyapunov_fn(target_lyapunov_params, next_obs, next_actions)

            # cost `c` is taken from the batch
            l_target = costs + self.gamma * (1 - dones) * l_next

            # Current L value
            l_current = next_lyapunov_fn(lyapunov_params, obs, actions)
            
            loss = ((l_current - l_target)**2).mean() * self.use_lyapunov
            return loss, {'lyapunov_loss': loss}

        (l_loss_val, l_metrics), l_grads = jax.value_and_grad(lyapunov_loss_fn, has_aux=True)(lyapunov_state.params)
        l_metrics['lyapunov_grad_norm'] = optax.global_norm(l_grads)  # raw gradient norm before clipping
        # Decide whether to apply the gradient based on the flag
        # If use_lyapunov=False, the gradient is 0 and new_lyapunov_state equals the old one
        new_lyapunov_state = lyapunov_state.apply_gradients(grads=l_grads)


        # --- Actor and Alpha update ---
        def actor_alpha_loss_fn(actor_params, alpha_params, lambda_params):


            dist_new = next_dist_fn(actor_params, obs)
            raw_actions_new = dist_new.sample(seed=actor_key)
            
            # --- Tanh correction applied to the critic target computation ---
            log_probs_new = apply_tanh_correction(dist_new, raw_actions_new)
            actions_new = jnp.tanh(raw_actions_new)

            q1, q2 = next_q_fn(new_critic_state.params, obs, actions_new)
            q = jnp.minimum(q1, q2)

            alpha_detached = jnp.exp(jax.lax.stop_gradient(alpha_params['log_alpha']))
            actor_loss_sac = (alpha_detached * jnp.squeeze(log_probs_new) - q).mean()


            alpha = jnp.exp(alpha_params['log_alpha'])
            log_probs_detached = jax.lax.stop_gradient(log_probs_new)
            alpha_loss = alpha_params['log_alpha'] * (-log_probs_detached.mean() - self.target_entropy)
            
            # b. Compute the Lyapunov-related loss terms
            # Compute L(s, a) and L(s, a')
            # L(s, a) uses the old action from the buffer
            edge_dist_next = next_dist_fn(actor_params, edge_next_obs)
            edge_raw_actions_next = edge_dist_next.sample(seed=actor_key)

            # --- Tanh correction applied to the critic target computation ---
            edge_log_probs_next = apply_tanh_correction(edge_dist_next, edge_raw_actions_next)
            edge_actions_next = jnp.tanh(edge_raw_actions_next)

            l_current_for_actor = jax.lax.stop_gradient(next_lyapunov_fn(new_lyapunov_state.params, edge_obs, edge_actions))
            # L(s, a') uses the new action generated by the actor
            l_next_for_actor = next_lyapunov_fn(new_lyapunov_state.params, edge_next_obs, edge_actions_next)

            # l_delta = E[L(s, a') - L(s, a) + alpha3 * c(s, a)]
            l_delta = (l_next_for_actor * jnp.squeeze(edge_next_obs.env_states.edge_mask) -
                      (l_current_for_actor - self.alpha3 * edge_costs) * jnp.squeeze(edge_obs.env_states.edge_mask)).mean() * use_edge_batch_flag
            # Change the constraint to l_delta <= epsilon: only penalize the violation beyond the margin, absorbing a small positive bias and removing the root cause of lambda windup
            # (the margin is multiplied by use_edge_batch_flag so l_delta_eff stays 0 when there is no edge data)
            l_delta_eff = l_delta - self.l_delta_margin * use_edge_batch_flag
            lambda_val  = jnp.clip(jnp.exp(lambda_params['log_lambda']), 0, self.lambda_clip_max)
            actor_loss_lyapunov = (jax.lax.stop_gradient(lambda_val) * l_delta_eff) * self.use_lyapunov * use_edge_batch_flag

            # d. Compute the Lambda loss (based on the margin-adjusted violation)

            lambda_loss = - lambda_params['log_lambda'] * jax.lax.stop_gradient(l_delta_eff) * self.use_lyapunov * use_edge_batch_flag
            
            actor_loss = actor_loss_sac + self.lyapunov_loss_coeff * actor_loss_lyapunov 
            total_loss = actor_loss + alpha_loss + lambda_loss
            return total_loss, (actor_loss, alpha_loss, lambda_loss, {'actor_loss': actor_loss, 
                                                         'alpha_loss': alpha_loss, 
                                                         'alpha': alpha, 
                                                         'entropy': -log_probs_detached.mean(),
                                                         'lambda_loss': lambda_loss, 
                                                         'lambda': lambda_val,
                                                         'l_delta': l_delta,
                                                         'l_delta_eff': l_delta_eff})

        grad_fn = jax.value_and_grad(actor_alpha_loss_fn, argnums=(0, 1, 2), has_aux=True)
        ((_, (actor_loss_val, _, _, actor_alpha_metrics)), 
         (actor_grads, alpha_grads, lambda_grads)) = grad_fn(actor_state.params, alpha_state.params, lambda_state.params)

        # Raw gradient norms before clipping, for monitoring
        actor_alpha_metrics['actor_grad_norm'] = optax.global_norm(actor_grads)
        actor_alpha_metrics['alpha_grad_norm'] = optax.global_norm(alpha_grads)
        actor_alpha_metrics['lambda_grad_norm'] = optax.global_norm(lambda_grads)

        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_lambda_state = lambda_state.apply_gradients(grads=lambda_grads)
        
        # --- Soft update of the target Critic ---
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
    
    def save_agent_states(self, save_path, step, prefix="best_"):
        """Saves all TrainState objects."""
        
        save_data = {
            'actor': self.actor_state,
            'critic': self.critic_state,
            'alpha': self.alpha_state
        }
        # checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="best_")
        checkpoints.save_checkpoint(ckpt_dir=save_path, target=save_data, step=step, prefix=prefix,keep=100,overwrite=False)
        print(f"Agent states saved to directory: {save_path}")

    def load_agent_states(self, load_path):
        """Loads all TrainState objects from a directory."""
        
        # Create a template to restore the states
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
        self.target_critic_params = self.critic_state.params # sync the target network
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
    Efficiently run multiple evaluation episodes in parallel using JAX vmap and scan.

    Args:
        agent: SACAgent instance (static argument, provides methods)
        eval_env: JAX-compatible environment instance (static argument)
        max_steps: maximum number of steps per episode (static argument)
        actor_params: trained Actor network parameters (dynamic argument)
        seed: initial random seed
        eval_episodes: number of episodes to run in parallel

    Returns:
        A JAX array containing the final total reward of each episode.
    """

    # --- 1. Define the core logic for running a *single* episode (rollout) ---
    def rollout_single_episode(key):

        # a. Reset the environment
        reset_key, rollout_key = jr.split(key)
        initial_graph = eval_env.reset(reset_key)

        # b. Define the single-step loop body (scan body)
        def step_fn(carry, _):
            # carry holds (current graph, cumulative reward, key, done flag)
            prev_graph, cumulative_reward, key, done_flag = carry

            # --- Use lax.cond to emulate early termination ---
            def do_step():
                a_key, next_key = jax.random.split(key)
                action = agent.select_action(a_key, actor_params, prev_graph, deterministic=True)
                next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
                new_cumulative_reward = cumulative_reward + reward
                return next_graph, new_cumulative_reward, next_key, done

            def skip_step():
                return prev_graph, cumulative_reward, key, done_flag

            next_graph, new_cumulative_reward, new_key, current_done_signal = jax.lax.cond(
                done_flag,
                skip_step,
                do_step
            )

            new_done_flag = jnp.logical_or(done_flag, current_done_signal)
            return (next_graph, new_cumulative_reward, new_key, new_done_flag), None

        # c. Set the initial scan state
        initial_carry = (initial_graph, 0.0, rollout_key, jnp.array(False))

        # d. Run the whole episode with lax.scan
        final_carry, _ = jax.lax.scan(
            step_fn,
            initial_carry,
            None,
            length=max_steps
        )

        # e. Extract the final total reward
        final_graph, final_reward, _, _ = final_carry
        successful_flag = jnp.where(final_graph.env_states.dist2tgt<= eval_env._params["car_radius"]*1.5,1,0)
        safe_flag = jnp.where(final_graph.env_states.timestep>=max_steps,1,0)
        return successful_flag, safe_flag, final_reward
        
    # --- 2. Parallelize the rollout function with vmap ---
    # a. Create a separate key for each parallel episode: episode i uses PRNGKey(seed+i),
    #    consistent with the detailed evaluation loop, so remembering the seed reproduces a single episode
    keys = jax.vmap(jax.random.PRNGKey)(seed + jnp.arange(eval_episodes))

    # b. vmap applies rollout_single_episode to all keys in parallel
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
        self.log_dir = os.path.abspath(log_dir) # ensure it is an absolute path
        self.seed = seed
        self.action_low, self.action_high = env.action_lim()
        # Static parameter check
        if GLACTrainer._check_params(params):
            self.params = params

        # Extract the configuration from the parameters
        self.edge_coeff = params['edge_coeff']
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
        # Create the model save directory
        if self.save_log:
            self.model_dir = os.path.join(self.log_dir, 'models')
            os.makedirs(self.model_dir, exist_ok=True)
        
        wandb.login()
        wandb.init(
            name=params['run_name'], 
            project=params['project_name'], 
            dir=self.log_dir,
            config=configs
        )
        self.key = jax.random.PRNGKey(seed)
        _, self.key = jax.random.split(self.key)
        self.env_model_error = 0
        self.model_steps = 100
        self.update_steps = 0
        self.best_eval_reward = -np.inf
        self.best_successful_rate = 0

    @staticmethod
    def _check_params(params: dict) -> bool:
        # Keep the original parameter-checking logic, adapted for SAC parameters
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
        
        # a. Reset the environment
        reset_key, rollout_key = jr.split(key)
        initial_graph = env.reset(reset_key)
        initial_N = -1
        # b. Define the single-step loop body (scan body)
        def step_fn(carry, _):
            # carry holds (current graph, cumulative reward, cumulative cost, key, done flag, prev_N)
            prev_graph, cumulative_reward, cumulative_cost, key, done_flag, prev_N = carry

            # --- Use lax.cond to emulate early termination ---
            def do_step():
                # If not done yet, take a normal step
                a_next_key, next_key = jax.random.split(key)
                action = agent.select_action(a_next_key, actor_params, prev_graph, deterministic=False)
                next_graph, reward, cost, done, info = env.step(prev_graph, action)
                current_N =  jnp.where(
                next_graph.env_states.min_dist2obs < env._params["car_radius"] * self.edge_coeff,
                next_graph.env_states.timestep,
                prev_N #
                )
                transition = (prev_graph,action,reward,cost,done, next_graph)
                new_cumulative_reward = cumulative_reward + reward
                new_cumulative_cost = cumulative_cost + cost
                return next_graph, action, new_cumulative_reward, new_cumulative_cost, next_key, done, transition, jnp.squeeze(current_N), info

            def skip_step():
                # If already done, the state and reward no longer change
                action = jnp.zeros((env.num_agents, env.action_dim))
                next_graph, reward, cost, done, info = env.step(prev_graph, action)
                transition = (prev_graph,action, jnp.array(0.0), jnp.array(0.0), done_flag, prev_graph)
                return prev_graph, action, cumulative_reward, cumulative_cost, key, done_flag, transition, prev_N, info

            # Choose which branch to run based on the previous done_flag
            next_graph, action, new_cumulative_reward, new_cumulative_cost, new_key, current_done_signal, current_transition, current_N, info = jax.lax.cond(
                done_flag,
                skip_step,
                do_step
            )

            # Update done_flag: once True, stay True
            new_done_flag = current_done_signal

            # Return the carry for the next iteration
            return (next_graph, new_cumulative_reward, new_cumulative_cost, new_key, new_done_flag, current_N), (current_transition, info)

        # c. Set the initial scan state
        initial_carry = (initial_graph, 0.0, 0.0, rollout_key, jnp.array(False), initial_N)

        # d. Run the whole episode with lax.scan
        final_carry, (all_transition, infos) = jax.lax.scan(
            step_fn,
            initial_carry,
            None,
            length=max_steps
        )

        # e. Extract the final total reward
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
            # Update the world model
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
            # Print and log here
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
            episode_transitions = (graph,action,reward,cost,done, next_graph)

            
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
                    if self.update_steps % 100 == 0: # reduce logging frequency
                        wandb.log({f"train/{k}": v.item() for k, v in update_info.items()}, step=current_step)
                    self.update_steps += 1

            # --- 3. Evaluation and model saving ---
            if collect_time % self.eval_interval == 0:
                # Call the parallel evaluation function
                all_successful_flag, all_safe_flag, all_episode_rewards = \
                run_parallel_evaluation(agent=self.agent,
                eval_env=self.env_test,
                max_steps=self.env_test.max_step,
                eval_episodes=self.eval_epi,
                actor_params=self.agent.actor_state.params, # pass in the latest parameters
                seed=self.seed)
                all_episode_rewards_np = np.array(all_episode_rewards)
                # Compute the mean
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
                    # Save the best model
                    if eval_reward > self.best_eval_reward or eval_successful_rate >= (self.best_successful_rate +0.02):#
                        self.best_eval_reward = eval_reward
                        self.best_successful_rate = eval_successful_rate
                        success_text = f"Success Rate : {eval_successful_rate*100} %\n"
                        eval_dir = os.path.join(self.model_dir, f"eval_train_best")
                        os.makedirs(eval_dir, exist_ok=True)
                        # Open the file and write the content (create it if it does not exist)
                        txt_path = os.path.join(eval_dir, f"output.txt")
                        with open(txt_path, "w", encoding="utf-8") as file:
                            file.writelines([success_text])
                        tqdm.write(f"New best model found! Saving...")
                        #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="best_")
                        self.agent.save_agent_states(self.model_dir, current_step, prefix="best_")
                    
            # Periodic save
            if collect_time % self.save_interval == 0 and self.save_log:
                tqdm.write(f"Saving interval checkpoint...")
                self.agent.save_agent_states(self.model_dir, current_step, prefix="checkpoint_")
                #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), current_step, prefix="checkpoint_")
            collect_time += 1
            pbar.update(episode_length)

        # Save the final model
        if self.save_log:
            print("Training finished. Saving final model.")
            self.agent.save_agent_states(self.model_dir, current_step, prefix="final_")
            #checkpoints.save_checkpoint(self.model_dir, self.agent.save_agent_states(), self.total_steps, prefix="final_")

        wandb.finish()