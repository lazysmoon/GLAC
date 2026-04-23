import os
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flax.training import checkpoints
import functools as ft
import jax.tree_util as jtu
import jax.random as jr
from glac.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap, tree_merge
from glac.rl_agent.data import Rollout
from glac.custom_envs.base import RolloutResult
from glac.rl_agent.glac import GLACAgent
from glac.custom_envs import make_env
from tqdm import tqdm
import sys, pickle

def is_debug_mode():
    """Check if running in debug mode."""
    return sys.gettrace() is not None

def load_config(args) -> tuple[dict]:
    env_params={
            'collision_penalty': args.collision_penalty,
            'success_reward': args.success_reward,
            'reach_reward': args.reach_reward,
            'correction_cost_dist': args.correction_cost_dist,
            'w_delta1': args.w_delta1,
            'w_delta2': args.w_delta2,
            'danger_penalty_coeff': args.danger_penalty_coeff,
            'potential_obs_reward_coeff': args.potential_obs_reward_coeff,
            'tgt_reward_coeff': args.tgt_reward_coeff,
            'reward_scale': args.reward_scale,
            'cost': args.cost,
            'cost_coeff': args.cost_coeff,
            'cost_dist': args.cost_dist,
            'cost_obs_dist': args.cost_obs_dist,
            }
    return env_params

# --- 1. Define core rollout logic for a single episode ---
@ft.partial(jax.jit, static_argnames=('agent', 'eval_env','max_steps'))
def rollout_single_episode(agent, 
    eval_env, 
    max_steps: int,
    key):
    
    reset_key, rollout_key = jr.split(key)
    initial_graph = eval_env.reset(reset_key)
    
    def step_fn(carry, _):
        prev_graph, cumulative_reward, key, done_flag = carry
        
        def do_step():
            a_key, next_key  = jax.random.split(key)
            action = agent.select_action(a_key, agent.actor_state.params, prev_graph, deterministic=True)

            next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
            transition = (prev_graph,action,reward,cost,done, next_graph)
            new_cumulative_reward = cumulative_reward + reward
            return next_graph, new_cumulative_reward, next_key, done, transition, info

        def skip_step():
            action = agent.select_action(key, agent.actor_state.params, prev_graph, deterministic=True)
            next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
            transition = (prev_graph,action, jnp.array(0.0), jnp.array(0.0), done_flag, prev_graph)
            return prev_graph, cumulative_reward, key, done_flag, transition, info

        # Choose branch based on previous done_flag
        next_graph, new_cumulative_reward, new_key, current_done_signal, current_transition, info = jax.lax.cond(
            done_flag,
            skip_step,
            do_step
        )
        
        # Update done_flag: once True, stays True
        new_done_flag = jnp.logical_or(done_flag, current_done_signal)
        
        # Return carry for the next iteration
        return (next_graph, new_cumulative_reward, new_key, new_done_flag), (current_transition, info)

    # c. Set initial carry for scan
    initial_carry = (initial_graph, 0.0, rollout_key, jnp.array(False))
    
    # d. Run the entire episode using lax.scan
    final_carry, (all_transition, infos) = jax.lax.scan(
        step_fn, 
        initial_carry, 
        None, 
        length=max_steps
    )
    
    # e. Extract final total reward
    _, final_reward, _, info= final_carry

    return final_reward, all_transition, infos
def test_model(args):
    print(f"> Running evaluation with args: {args}")
    env_params = load_config(args)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
    # --- 1. Set up environment and agent ---
    # Create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step = args.max_step,
        r_c_params=env_params

    )

    # Initialize agent (must match training structure)
    agent = GLACAgent(
        env=env,
        n_agents=env.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        seed=args.seed,
        # Other hyperparameters can be set arbitrarily
    )
    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")

        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    # --- 2. Load trained model ---
    # Use checkpoints.latest_checkpoint to find the latest best model
    prefix = args.prefix
    load_path = checkpoints.latest_checkpoint(ckpt_dir=args.model_dir, prefix=prefix)
    if not load_path:
        # If best not found, try final
        load_path = checkpoints.latest_checkpoint(ckpt_dir=args.model_dir, prefix="checkpoint")
    if not load_path:
        raise FileNotFoundError(f"No checkpoint found in directory: {args.model_dir}")
    load_path = os.path.abspath(load_path)
    print(f"Loading agent states from: {load_path}")
    agent.load_agent_states(load_path)
    save_dir = os.path.join(args.model_dir, f"eval_obs{args.obs}_{prefix}")
    os.makedirs(save_dir, exist_ok=True)
    episodes = []
    episodes_returns = []
    episode_dist2tgt = []
    # --- 4. Run evaluation loop ---
    # Collect data following the GlACTrainer structure
    pbar = tqdm(total=args.epi, desc="Evaluating Episodes")
    success_time = 0
    safe_time = 0
    for i in range(args.epi):
        keys = jax.random.PRNGKey(args.seed+i)
        episodes_return, all_transitions, infos = rollout_single_episode( agent, env, env.max_step, keys)
        episodes_returns.append(episodes_return)
        (graph,action,reward,cost,done, next_graph) = all_transitions
        done_indices = np.where(done)[0]
        done_index = done_indices[0]
        infos_np = jtu.tree_map(np.asarray, infos)
        dist2tgt = infos_np['dist2tgt'][done_index]
        episode_dist2tgt.append(dist2tgt)
        # T_... data has length T
        episode_transitions = jtu.tree_map(
            lambda x: x[0:done_index+1],
            (action, reward,cost, done)
        )
        
        # Tp1_graph data has length T+1
        episode_graph = jtu.tree_map(
            lambda x: x[0:done_index+1],  # Note: end+1 inclusive
            graph
        )
        episodes.append(RolloutResult(
                Tp1_graph=episode_graph,
                T_action=episode_transitions[0],
                T_reward=episode_transitions[1],
                T_cost=episode_transitions[2],
                T_done=episode_transitions[3],
                T_info=None
            ))
        episode_verbose = (f"Episode {i+1}: episodes_return={episodes_return:.2f}, Episode_Length={done_index}, dist2tgt = {dist2tgt}")
        if dist2tgt < 0.1:
            success_time += 1
        if done_index+1>=args.max_step:
            safe_time += 1
        tqdm.write(episode_verbose)
        pbar.update(1)
        pass
     # --- 5. Print summary ---
    mean_return = np.mean(episodes_returns)
    std_return = np.std(episodes_returns)
    safe_rate = safe_time/args.epi*100
    success_rate = success_time/args.epi*100
    success_text = f"Success times: {success_time}. Success Rate : {success_rate:.2f} %\n"
    safe_text = f"Safe times: {safe_time}. Safe Rate : {safe_rate:.2f} %\n"
    print("\n----------------------------------------------------")
    print(f"Evaluation over {args.epi} episodes:")
    print(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    print(success_text,safe_text)
    print("----------------------------------------------------")
    # Save evaluation statistics
    # Write to file (create if not exists)
    txt_path = os.path.join(save_dir, f"output.txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        file.writelines([f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}\n", success_text, safe_text])
    print("Results successfully written to output.txt")

    idx_for_vd = np.random.randint(0,len(episodes), size=10)
    idx_set = set(idx_for_vd)
    for i, episode_rollout in enumerate(episodes):
           if i in idx_set:
           #if i < 10:
                # rollout_np = jtu.tree_map(lambda x: np.array(x) if hasattr(x, '__array__') else x, episode_rollout)
                # save_pkl_path = os.path.join(save_dir, f"{args.prefix}_seed10.pkl")
                # print(f"Saving rollout data to {save_pkl_path}...")
                # with open(save_pkl_path, 'wb') as f:
                #     pickle.dump(rollout_np, f)
                video_path = os.path.join(save_dir, f"ep_{i+1}_return_{episodes_returns[i]:.2f}.mp4")
                png_path = os.path.join(save_dir, f"ep_{i+1}_return_{episodes_returns[i]:.2f}.png")
                # Compute is_unsafe for this episode
                Ta_is_unsafe = is_unsafe_fn(episode_rollout.Tp1_graph)
                # Call rendering function
                env.render_trajectory(rollout=episode_rollout,save_path=png_path, dpi=300)
                print(f"Rendering video for episode {i+1} to {video_path} ...")
                env.render_video(rollout =episode_rollout, video_path=video_path, Ta_is_unsafe = Ta_is_unsafe, dpi=300)             
                pass

def main():
    parser = argparse.ArgumentParser()
    # --- Core arguments ---
    parser.add_argument("--model_dir", type=str,   default= './pretrain', help="Directory where the trained models are saved.")
    parser.add_argument("--env", type=str, default='Second_order', help="Name of the environment.")
    parser.add_argument("--prefix", type=str, default= 'checkpoint', help="Name of the model.")
    # --- Evaluation arguments ---
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation.")
    parser.add_argument("--epi", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--max_step", type=int, default=320, help="Maximum steps per episode.")
    

    # Reward/cost shaping arguments
    parser.add_argument("--collision_penalty", type=float, default=-300)
    parser.add_argument("--success_reward", type=float, default=100)
    parser.add_argument("--reach_reward", type=float, default=5)
    parser.add_argument("--correction_cost_dist", type=float, default=0.05)
    parser.add_argument("--w_delta1", type=float, default=1)
    parser.add_argument("--w_delta2", type=float, default=5)
    parser.add_argument("--danger_penalty_coeff", type=float, default=40)
    parser.add_argument("--potential_obs_reward_coeff", type=float, default=40)
    parser.add_argument("--tgt_reward_coeff", type=float, default=100)
    parser.add_argument("--reward_scale", type=float, default=5)
    parser.add_argument("--cost", type=float, default=1)
    parser.add_argument("--cost_dist", type=float, default=0.06)
    parser.add_argument("--cost_obs_dist", type=float, default=2)
    parser.add_argument("--cost_coeff", type=float, default=10)

    # --- Environment-specific parameters (should match training) ---
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents.")
    parser.add_argument("--area-size", type=float, default=4.0, help="Size of the environment area.")
    parser.add_argument("--obs", type=int, default=16) #12 16 20
    parser.add_argument("--n-rays", type=int, default=32)
    
    # --- Optional features ---
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--no-video", action="store_true", help="Do not generate and save videos.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved videos.")
    parser.add_argument("--debug", action="store_true", default=is_debug_mode())
    args = parser.parse_args()
    test_model(args)


if __name__ == "__main__":
    main()