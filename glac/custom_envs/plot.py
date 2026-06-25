import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union

from ..utils.utils import centered_norm
from ..utils.typing import EdgeIndex, Pos2d, Pos3d, Array
from ..utils.utils import merge01, tree_index, MutablePatchCollection, save_anim
from .obstacle import Cuboid, Sphere, Obstacle, Rectangle, GeneralObstacle
from .base import RolloutResult
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize



def get_obs_collection(
        obstacles: Obstacle, color: str, alpha: float
):
    n_obs = len(obstacles.center)
    patches = []
    
    # Get the type array. Legacy Rectangle classes may lack a `type` field; default to rectangle.
    obs_types = getattr(obstacles, 'type', None)

    for i in range(n_obs):
        # Decide whether it is a circle (type == 1)
        # Note: obs_types[i] may be a JAX array; comparisons convert automatically,
        # but to be safe we check its value only when obs_types exists.
        is_circle = False
        if obs_types is not None:
            # Assume 1 means circle (CIRCLE)
            if obs_types[i] == 1:
                is_circle = True

        if is_circle:
            # --- Draw a circle ---
            # center: (x, y), radius: float
            xy = (obstacles.center[i][0], obstacles.center[i][1])
            r = obstacles.radius[i]
            patches.append(Circle(xy, radius=r))
        else:
            # --- Draw a rectangle/polygon ---
            # points: (4, 2)
            poly_points = obstacles.points[i]
            patches.append(Polygon(poly_points, closed=True))

    obs_col = PatchCollection(patches, facecolor=color, alpha=alpha, zorder=99)

    return obs_col


def render_video(
        rollout: RolloutResult,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_rays: int,
        r: float,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        **kwargs
):
    assert dim == 2 or dim == 3

    # set up visualization option
    if dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # plot the first frame
    T_graph = rollout.Tp1_graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#404040"
    edge_goal_color = goal_color

    # plot obstacles
    obs = graph0.env_states.obstacle
    ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))

    # plot agents
    n_hits = n_agent * n_rays
    n_color = [agent_color] * n_agent + [goal_color] * n_agent
    n_pos = graph0.states[:n_agent * 2, :dim]
    n_radius = np.array([r] * n_agent * 2)
    if dim == 2:
        # agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
        #                for ii in range(n_agent * 2)]
        # agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        # ax.add_collection(agent_col)
        agent_list = [] # holds all the shape objects
        for ii in range(n_agent * 2):
            if ii < n_agent:
                # --- First half are agents: draw circles ---
                # The center is simply n_pos[ii]
                patch = plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
            else:
                # --- Second half are goals: draw squares ---
                # Rectangle's (x, y) is the bottom-left corner, so subtract radius r to center it
                bottom_left_x = n_pos[ii, 0] - r
                bottom_left_y = n_pos[ii, 1] - r
                side_length = r * 2  # side length equals the diameter
                patch = plt.Rectangle((bottom_left_x, bottom_left_y), side_length, side_length,
                                  color=n_color[ii], linewidth=0.0)
            agent_list.append(patch)

        # Note: renamed from agent_circs to agent_list since it now mixes circles and squares
        agent_col = MutablePatchCollection([i for i in reversed(agent_list)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                                s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent * 2 + n_hits, :dim]
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent * 2 + n_hits, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent * 2)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 2:
        cost_text = ax.text(0.02, 1.04, "dist2obs: 1.0, dist2tgt: 1.0, Reward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.04, "dist2obs: 1.0, dist2tgt: 1.0, Reward: 1.0", va="bottom", **text_font_opts)

    # text for safety
    # safe_text = []
    # if Ta_is_unsafe is not None:
    #     if dim == 2:
    #         safe_text = [ax.text(0.02, 1.00, "Unsafe: {}", va="bottom", **text_font_opts)]
    #     else:
    #         safe_text = [ax.text2D(0.02, 1.00, "Unsafe: {}", va="bottom", **text_font_opts)]

    # text for time step
    if dim == 2:
        kk_text = ax.text(0.99, 0.99, "kk=0", va="top", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 0.99, "kk=0", va="top", ha="right", **text_font_opts)


    # init function for animation
    def init_fn() -> list[plt.Artist]:
        return [agent_col, edge_col, cost_text, kk_text] #*agent_labels, *safe_text, *cnt_col,

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]

        # update agent positions
        if dim == 2:
            for ii in range(n_agent * 2): 
                if ii < n_agent:
                    # --- Agent (circle): update the center ---
                    agent_list[ii].set_center(tuple(n_pos_t[ii]))
                else:
                    # --- Goal (square): update the bottom-left corner ---
                    # Again subtract r to keep it centered
                    new_x = n_pos_t[ii, 0] - r
                    new_y = n_pos_t[ii, 1] - r
                    agent_list[ii].set_xy((new_x, new_y))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent * 2, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent * 2, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent * 2 + n_hits, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent * 2)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # # update agent labels
        # for ii in range(n_agent):
        #     if dim == 2:
        #         agent_labels[ii].set_position(n_pos_t[ii])
        #     else:
        #         text_pos = proj3d.proj_transform(n_pos_t[ii, 0], n_pos_t[ii, 1], n_pos_t[ii, 2], ax.get_proj())[:2]
        #         agent_labels[ii].set_position(text_pos)

        # update cost and safe labels
        if kk < len(rollout.T_cost):
            dist2tgt = rollout.Tp1_graph.env_states.dist2tgt[kk][0]
            min_dist2obs = rollout.Tp1_graph.env_states.min_dist2obs[kk][0]
            cost_text.set_text("dist2obs: {:5.4f}, dist2tgt: {:5.4f}, Reward: {:5.4f}".format(min_dist2obs, dist2tgt, rollout.T_reward[kk]))
        else:
            cost_text.set_text("")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col,  cost_text, kk_text]#*agent_labels, *safe_text, *cnt_col_t,

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


def render_trajectory_with_radius(
        rollout: RolloutResult,
        save_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        r: float,
        dt: float = 0.03,
        dpi: int = 150,
        spacing: float = None,
        draw_heading: bool = True,
        circle_alpha: float = 0.85,
        cmap_name: str = 'viridis',
        **kwargs,
):
    """Plot a static trajectory with the car radius: draw a series of circles of radius
    ``r`` (the car's physical footprint) along the trajectory, spaced by arc length, with
    a time-gradient color and optional heading arrows. This visualizes the avoidance
    difficulty (how narrowly the car passes obstacles). Compared to
    :func:`render_trajectory`, it only adds the car-body circles.

    Args:
        r: car radius (car_radius), the circle radius.
        spacing: arc-length spacing between adjacent circle centers. Defaults to ``1.6 * r`` (adjacent circles slightly overlap) when None.
        draw_heading: whether to draw a short heading arrow on each circle (requires the state to contain the heading angle theta).
        circle_alpha: transparency of the car-body circles.
    """
    if dim != 2:
        raise NotImplementedError("The static trajectory plot only supports 2D environments")
    if spacing is None:
        spacing = 1.6 * r

    # 1. Canvas
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=dpi)
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    ax.set_aspect("equal")

    # 2. Extract the full state at all time steps (including the heading angle theta)
    T_graph = rollout.Tp1_graph
    total_steps = len(T_graph.n_node)
    all_states = []
    for kk in range(total_steps):
        graph = tree_index(T_graph, kk)
        all_states.append(np.asarray(graph.states[:n_agent]))
    all_states = np.array(all_states)          # (T, n_agent, state_dim)
    all_positions = all_states[:, :, :dim]      # (T, n_agent, 2)
    state_dim = all_states.shape[-1]

    # 3. Obstacles
    graph0 = tree_index(T_graph, 0)
    obs = graph0.env_states.obstacle
    try:
        obs_col = get_obs_collection(obs, color="k", alpha=1.0)
        obs_col.set_facecolor("#666666")
        obs_col.set_edgecolor('#202020')
        obs_col.set_linewidth(1)
        obs_col.set_alpha(0.8)
        obs_col.set_zorder(10)
        ax.add_collection(obs_col)
    except Exception:
        pass

    # 4. Color map (over time)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=total_steps * dt)
    lc = None
    special_blue = '#007bff'

    def arclength_sample_idx(traj):
        """Sample along the trajectory by ``spacing`` arc length; return time-step indices (including first and last)."""
        seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-9:
            return np.array([0, len(traj) - 1])
        targets = np.arange(0.0, total, spacing)
        idx = np.searchsorted(cum, targets)
        idx = np.clip(idx, 0, len(traj) - 1)
        idx = np.unique(np.concatenate([idx, [len(traj) - 1]]))
        return idx

    for i in range(n_agent):
        full_traj = all_positions[:, i, :]              # (T, 2)

        # --- 4a. Underlying thin gradient line, to keep trajectory continuity ---
        points = full_traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(len(segments)) * dt)
        lc.set_linewidth(1.2)
        lc.set_alpha(0.6)
        lc.set_zorder(4)
        ax.add_collection(lc)

        # --- 4b. Draw the car-radius circles along the trajectory (core) ---
        idxs = arclength_sample_idx(full_traj)
        for k in idxs:
            t = k * dt
            color = cmap(norm(t))
            circ = Circle(
                (full_traj[k, 0], full_traj[k, 1]), radius=r,
                facecolor=color, edgecolor='#202020', linewidth=0.6,
                alpha=circle_alpha, zorder=6,
            )
            ax.add_patch(circ)
            # Center point
            ax.scatter(full_traj[k, 0], full_traj[k, 1], c='#202020', s=3,
                       marker='o', zorder=8)
            # Short heading arrow
            if draw_heading and state_dim > 2:
                theta = all_states[k, i, 2]
                dx, dy = np.cos(theta) * r * 0.9, np.sin(theta) * r * 0.9
                ax.annotate(
                    "", xy=(full_traj[k, 0] + dx, full_traj[k, 1] + dy),
                    xytext=(full_traj[k, 0], full_traj[k, 1]),
                    arrowprops=dict(arrowstyle="-|>", color='#202020', lw=0.8,
                                    shrinkA=0, shrinkB=0), zorder=9,
                )

        # --- 4c. Start (circle) and end (square) ---
        ax.scatter(full_traj[0, 0], full_traj[0, 1], c=special_blue, s=60,
                   marker='o', edgecolors='white', linewidth=1.5, zorder=20,
                   label='Start' if i == 0 else "")
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], c=special_blue, s=80,
                   marker='s', edgecolors='white', linewidth=1.5, zorder=20,
                   label='Goal' if i == 0 else "")

    # 5. Colorbar
    if lc is not None:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('time (s)', fontsize=12)
        max_time = total_steps * dt
        cbar_ticks = np.linspace(0, max_time, 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in cbar_ticks])
        cbar.solids.set_edgecolor("face")
        cbar.outline.set_linewidth(1)

    plt.tight_layout()
    print(f"Saving trajectory-with-radius plot to {save_path}")
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)
