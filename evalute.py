import os
import time
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
from glac.rl_agent.glac import GLACAgent, run_parallel_evaluation
from glac.custom_envs import make_env
from tqdm import tqdm
import sys
import yaml
from flax.training.checkpoints import available_steps
plt.rcParams['font.cursive'] = ['Comic Sans MS']
matplotlib.rcParams['font.cursive'] = ['DejaVu Sans']  # key line: avoid failing to find 'cursive'

def is_debug_mode():
    """Check whether running in debug mode."""
    return sys.gettrace() is not None

def load_config(args) -> dict:
    # config.yaml is in the parent directory of model_dir (the run directory saved during training)
    run_dir = os.path.dirname(os.path.normpath(args.model_dir))
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.yaml found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.unsafe_load(f)
    cfg = config.get("command_line_args", config)
    print(f"> Loaded config from: {config_path}")

    # Override args with values from the config file (also sync environment-build params beyond env_params)
    for k in ("env", "dt", "num_agents", "n_rays"):
        if k in cfg:
            setattr(args, k, cfg[k])

    # obs / area_size: use the user's value if set explicitly (not None) in main, otherwise read from config
    for k in ("max_step", "obs", "area_size"):
        if getattr(args, k, None) is None and k in cfg:
            setattr(args, k, cfg[k])

    env_param_keys = [
        'collision_penalty', 'success_reward', 'reach_reward', 'correction_cost_dist',
        'w_delta1', 'w_delta2', 'warning_dist2obs', 'delta_action_scale',
        'danger_penalty_coeff', 'potential_obs_reward_coeff', 'tgt_reward_coeff',
        'reward_scale', 'cost', 'cost_coeff', 'cost_dist', 'cost_obs_dist',
    ]
    # Prefer values from the config file, falling back to the args defaults when missing
    env_params = {
        key: cfg[key] if key in cfg else getattr(args, key)
        for key in env_param_keys
    }
    return env_params

def get_checkpoint_path_by_step(ckpt_dir, prefix, step):
    """Find the checkpoint path for the given step; return None if it does not exist."""
    ckpt_path = os.path.join(ckpt_dir, f"{prefix}{step}")
    return ckpt_path if os.path.exists(ckpt_path) else None

def  is_checkpoint_dir(path):
    """Return True if `path` is itself an Orbax checkpoint directory (single saved checkpoint)."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "_CHECKPOINT_METADATA"))

# --- 1. Define the core logic for running a *single* episode (rollout) ---
@ft.partial(jax.jit, static_argnames=('agent', 'eval_env','max_steps'))
def rollout_single_episode(agent, 
    eval_env, 
    max_steps: int,
    key):
    
    # a. Reset the environment
    reset_key, rollout_key = jr.split(key)
    initial_graph = eval_env.reset(reset_key)

    # b. Define the single-step loop body (scan body)
    def step_fn(carry, _):
        # carry holds (current graph, cumulative reward, key, done flag)
        prev_graph, cumulative_reward, key, done_flag = carry

        # --- Use lax.cond to emulate early termination ---
        def do_step():
            # If not done yet, take a normal step
            a_key, next_key  = jax.random.split(key)
            action = agent.select_action(a_key, agent.actor_state.params, prev_graph, deterministic=True)

            next_graph, reward, cost, done, info = eval_env.step(prev_graph, action)
            transition = (prev_graph,action,reward,cost,done, next_graph)
            new_cumulative_reward = cumulative_reward + reward
            return next_graph, new_cumulative_reward, next_key, done, transition, info

        def skip_step():
            action = jnp.zeros((eval_env.num_agents, eval_env.action_dim))
            transition = (prev_graph, action, jnp.array(0.0), jnp.array(0.0), done_flag, prev_graph)
            info = {'dist2tgt': jnp.zeros(eval_env.num_agents)}
            return prev_graph, cumulative_reward, key, done_flag, transition, info

        # Choose which branch to run based on the previous done_flag
        next_graph, new_cumulative_reward, new_key, current_done_signal, current_transition, info = jax.lax.cond(
            done_flag,
            skip_step,
            do_step
        )

        # Update done_flag: once True, stay True
        new_done_flag = jnp.logical_or(done_flag, current_done_signal)

        # Return the carry for the next iteration
        return (next_graph, new_cumulative_reward, new_key, new_done_flag), (current_transition, info)

    # c. Set the initial scan state
    initial_carry = (initial_graph, 0.0, rollout_key, jnp.array(False))

    # d. Run the whole episode with lax.scan
    final_carry, (all_transition, infos) = jax.lax.scan(
        step_fn,
        initial_carry,
        None,
        length=max_steps
    )

    # e. Extract the final total reward
    _, final_reward, _, info= final_carry

    return final_reward, all_transition, infos
def test_model(args):
    print(f"> Running evaluation with args: {args}")
    env_params = load_config(args)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
    # --- 1. Set up the environment and agent ---
    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        dt=args.dt,
        r_c_params=env_params,
    )

    # Initialize the agent (structure must match training)
    agent = GLACAgent(
        env=env,
        n_agents=env.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        seed=args.seed,
        # other SAC hyperparameters can be set arbitrarily since they are not used
    )
    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")

        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    # --- 2. Load the trained model ---
    prefix = args.prefix

    if is_checkpoint_dir(args.model_dir):
        # model_dir is itself a checkpoint directory (e.g. the shipped pretrain/checkpoint):
        # load it directly and skip the step-sweep / step-lookup logic.
        load_path = os.path.abspath(args.model_dir)
        ckpt_name = os.path.basename(os.path.normpath(args.model_dir))
        save_dir = os.path.join(os.path.dirname(os.path.normpath(args.model_dir)),
                                f"eval_obs{args.obs}_{ckpt_name}")
    else:
        # If checkpoint_step is not specified, sweep all checkpoints in the directory and pick the best one via parallel evaluation
        if args.checkpoint_step is None:
            steps = available_steps(args.model_dir, prefix=prefix)
            if not steps:
                raise FileNotFoundError(f"No '{prefix}*' checkpoint found in: {args.model_dir}")
            best_success_rate, best_safe_rate = -0.1, -0.1
            best_success_safe_rate, best_safe_success_rate = -0.1, -0.1
            best_success_mean_return, best_safe_mean_return = -0.1, -0.1
            best_success_step, best_safe_step = steps[0], steps[0]
            print(f"Sweeping {len(steps)} checkpoints to find the best one...")
            for step in tqdm(steps, desc="Sweeping checkpoints"):
                load_path = get_checkpoint_path_by_step(args.model_dir, prefix, step)
                if not load_path:
                    continue
                agent.load_agent_states(os.path.abspath(load_path))
                all_successful_flag, all_safe_flag, all_return = run_parallel_evaluation(
                    agent=agent,
                    eval_env=env,
                    max_steps=env.max_step,
                    eval_episodes=args.epi,
                    actor_params=agent.actor_state.params,
                    seed=args.seed,
                )
                success_rate = all_successful_flag.mean().item()
                safe_rate = all_safe_flag.mean().item()
                mean_return = all_return.mean().item()
                tqdm.write(f"  step {step}: success_rate={success_rate*100:.2f}%, safe_rate={safe_rate*100:.2f}%")
                if success_rate >= best_success_rate:
                    best_success_rate, best_success_step = success_rate, step
                    best_success_safe_rate = safe_rate
                    best_success_mean_return  = mean_return
                if safe_rate >= best_safe_rate:
                    best_safe_rate, best_safe_step = safe_rate, step
                    best_safe_success_rate = success_rate
                    best_safe_mean_return = mean_return
            text = (f"best_success_rate: {best_success_rate*100:.2f}%, safe_rate: {best_success_safe_rate*100:.2f}%, mean_return: {best_success_mean_return:.2f} with step {best_success_step}\n"
                    f"best_safe_rate: {best_safe_rate*100:.2f}, success_rate: {best_safe_success_rate*100:.2f}%, mean_return: {best_safe_mean_return:.2f} with step {best_safe_step}\n")
            print(text)
            txt_path = os.path.join(args.model_dir, f"obs{args.obs}_size{args.area_size}_seed{args.seed}_all_{prefix}steps.txt")
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"Sweep results written to {txt_path}")
            # Use the step with the best success_rate for the detailed evaluation and rendering below
            chosen_step = best_success_step
        else:
            chosen_step = args.checkpoint_step

        load_path = get_checkpoint_path_by_step(args.model_dir, prefix, chosen_step)
        if not load_path:
            # Fallback: find the latest checkpoint under this prefix
            load_path = checkpoints.latest_checkpoint(ckpt_dir=args.model_dir, prefix=prefix)
        if not load_path:
            raise FileNotFoundError(f"No checkpoint found in directory: {args.model_dir}")
        load_path = os.path.abspath(load_path)
        save_dir = os.path.join(args.model_dir, f"eval_obs{args.obs}_{prefix}{chosen_step}")

    print(f"Loading agent states from: {load_path}")
    agent.load_agent_states(load_path)
    os.makedirs(save_dir, exist_ok=True)

    episodes = []
    episodes_returns = []
    episode_dist2tgt = []
    # --- 4. Run the evaluation loop ---
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
            lambda x: x[0:done_index+1], # note the end+1 here
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
        if dist2tgt < env._params["car_radius"] *1.5:
            success_time += 1
        if done_index+1>=args.max_step:
            safe_time += 1
        tqdm.write(episode_verbose)
        pbar.update(1)
        pass
   
    ##--- 6. Print the summary ---
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
    # Save the statistics
    # Open the file and write the content (create it if it does not exist)
    txt_path = os.path.join(save_dir, f"output.txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        file.writelines([f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}\n", success_text, safe_text])
    print("Successfully wrote the string to output.txt")

def main():
    parser = argparse.ArgumentParser()
    # --- Core parameters ---
    model_dir = './pretrain/checkpoint'
    parser.add_argument("--model_dir", type=str,   default= model_dir, help="Directory where the trained model is saved (an Orbax checkpoint dir, or a dir of checkpoint_<step> folders).")
    parser.add_argument("--env", type=str, default='DubinsCar', help="Name of the environment.")
    prefix = "checkpoint_"
    parser.add_argument("--prefix", type=str, default=prefix, help="Checkpoint name prefix (used only when model_dir contains multiple checkpoints).")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step to evaluate; None means auto-sweep for the best one.")
    # --- Evaluation parameters ---
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation.")
    parser.add_argument("--epi", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--max_step", type=int, default=256, help="Maximum steps per episode.")
    parser.add_argument("--dt", type=float, default=0.02)

    # --- Environment-specific parameters (must match training) ---
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents.")
    parser.add_argument("--area-size", type=float, default=None, help="Size of the environment area. Reads config.yaml when None.")
    parser.add_argument("--obs", type=int, default=6, help="Number of obstacles. Reads config.yaml when None.")
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