"""Waypoint-policy training with a differentiable MPC controller.

Training loop (all steps live on CUDA and keep gradients end-to-end):

    depth (CUDA renderer) ──► waypoint policy (model_mpc.Model)
                                  │  waypoints + desired speed
                                  ▼
                      differentiable MPC (mpc.DifferentiableWaypointMPC)
                                  │  command (v, omega)
                                  ▼
                    motor lag filter ──► CUDA diff-drive dynamics (env_cuda.Env)
                                  │
                                  ▼
                    loss (mirrors train.py) ──► BTTT backprop

Usage:
    python3 train_mpc.py @configs/train_param.args
    # CLI overrides: python3 train_mpc.py @configs/train_param.args --batch_size 32
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env_cuda import Env
from model_mpc import Model
from mpc import (
    DifferentiableWaypointMPC,
    depth_to_local_obstacle_points,
    estimate_emergency_risk,
    estimate_local_obstacle_velocity,
)

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'configs', 'train_param.args')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')


class ConfigArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that supports '#'-comments and inline values in @files."""

    def convert_arg_line_to_args(self, arg_line):
        line = arg_line.split('#', 1)[0].strip()
        return line.split() if line else []


def parse_args():
    parser = ConfigArgumentParser(
        fromfile_prefix_chars='@',
        description='Diff-Wheelbot waypoint policy + MPC training',
    )

    # --- basic training ---
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=120)
    parser.add_argument('--grad_decay', type=float, default=0.8)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
    parser.add_argument('--control_hz', type=float, default=15.0)
    parser.add_argument('--dt_noise_std', type=float, default=0.005)
    parser.add_argument('--motor_rate_min', type=float, default=3.0)
    parser.add_argument('--motor_rate_max', type=float, default=8.0)
    parser.add_argument('--max_action_delay_steps', type=int, default=0)
    parser.add_argument('--exec_v_scale_std', type=float, default=0.0)
    parser.add_argument('--exec_w_scale_std', type=float, default=0.0)
    parser.add_argument('--wheel_bias_std', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    # --- checkpoint / logging ---
    parser.add_argument('--save_dir', default='save')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument(
        '--run_name', default=None,
        help='Optional run-directory name; defaults to seed + timestamp.')
    parser.add_argument('--ckpt_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--plot_interval', type=int, default=1000)
    parser.add_argument('--resume', default=None)

    # --- loss weights (aligned with train.py) ---
    parser.add_argument('--coef_pos', type=float, default=1.0)
    parser.add_argument('--coef_v', type=float, default=1.0)
    parser.add_argument('--coef_heading', type=float, default=0.5)
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.0)
    parser.add_argument('--coef_collide', type=float, default=3.5)
    parser.add_argument('--coef_smooth', type=float, default=0.1)
    parser.add_argument('--coef_bias', type=float, default=0.5)
    parser.add_argument('--coef_energy', type=float, default=0.05)
    parser.add_argument('--avoid_safe_distance', type=float, default=1.0)
    parser.add_argument('--approach_speed_gain', type=float, default=1.0)
    parser.add_argument('--approach_speed_max', type=float, default=4.0)
    parser.add_argument('--collision_soft_clearance', type=float, default=0.30)
    parser.add_argument('--speed_distance_gain', type=float, default=0.5)
    parser.add_argument('--speed_min_scale', type=float, default=0.15)
    parser.add_argument('--speed_clearance_min', type=float, default=0.05)
    parser.add_argument('--speed_clearance_full', type=float, default=0.65)

    # --- waypoint policy (model_mpc.Model) ---
    parser.add_argument('--num_waypoints', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=192)
    parser.add_argument('--max_forward_step', type=float, default=1.5)
    parser.add_argument('--max_lateral_step', type=float, default=1.0)
    parser.add_argument('--max_speed', type=float, default=4.0)
    parser.add_argument('--max_omega', type=float, default=3.0)
    parser.add_argument('--initial_desired_speed', type=float, default=3.2)
    parser.add_argument('--min_desired_speed', type=float, default=0.15)

    # --- MPC (mpc.DifferentiableWaypointMPC) ---
    parser.add_argument('--mpc_horizon', type=int, default=12)
    parser.add_argument('--mpc_control_lookahead', type=int, default=3)
    parser.add_argument('--mpc_max_acc_v', type=float, default=8.0)
    parser.add_argument('--mpc_max_acc_omega', type=float, default=8.0)
    parser.add_argument('--mpc_max_lateral_acc', type=float, default=8.0)
    parser.add_argument('--mpc_track_weight', type=float, default=8.0)
    parser.add_argument('--mpc_smooth_weight', type=float, default=30.0)
    parser.add_argument('--mpc_initial_velocity_weight', type=float, default=4.0)
    parser.add_argument('--mpc_obstacle_clearance', type=float, default=0.30)
    parser.add_argument('--mpc_obstacle_temperature', type=float, default=0.15)
    parser.add_argument('--mpc_obstacle_refine_steps', type=int, default=2)
    parser.add_argument('--mpc_perception_safety', type=str2bool, default=True)

    # --- perception processing (mpc.py helpers) ---
    parser.add_argument('--obstacle_num_points', type=int, default=16)
    parser.add_argument('--obstacle_height_fraction', type=float, default=0.4)
    parser.add_argument('--obstacle_depth_quantile', type=float, default=0.1)
    parser.add_argument('--obstacle_min_range', type=float, default=0.2)
    parser.add_argument('--obstacle_max_range', type=float, default=6.0)
    parser.add_argument('--emergency_distance', type=float, default=1.5)
    parser.add_argument('--emergency_ttc', type=float, default=0.8)

    # --- environment ---
    parser.add_argument('--env_width', type=int, default=64)
    parser.add_argument('--env_height', type=int, default=48)
    parser.add_argument('--diff_nearest_pt', type=str2bool, default=True)
    parser.add_argument('--map_size', type=float, default=20.0)
    parser.add_argument('--num_cyl', type=int, default=25)
    parser.add_argument('--num_balls', type=int, default=15)
    parser.add_argument('--num_vox', type=int, default=15)
    parser.add_argument('--robot_radius', type=float, default=0.15)
    parser.add_argument('--cyl_radius_min', type=float, default=0.2)
    parser.add_argument('--cyl_radius_max', type=float, default=0.5)
    parser.add_argument('--ball_radius_min', type=float, default=0.2)
    parser.add_argument('--ball_radius_max', type=float, default=0.4)
    parser.add_argument('--ball_radius_floor', type=float, default=0.0)
    parser.add_argument('--randomize_start_goal', type=str2bool, default=True)
    parser.add_argument('--protected_zone_radius', type=float, default=2.0)
    parser.add_argument('--obstacle_min_surface_gap', type=float, default=0.0)
    parser.add_argument('--obstacle_resample_attempts', type=int, default=128)
    parser.add_argument('--obstacle_scene_restarts', type=int, default=8)
    parser.add_argument(
        '--obstacle_layout', choices=('nonoverlap', 'stratified'),
        default='nonoverlap')
    parser.add_argument('--obstacle_grid_jitter', type=float, default=0.75)
    parser.add_argument(
        '--obstacle_candidate_multiplier', type=float, default=2.0)
    parser.add_argument('--dynamic_obstacle_scene_prob', type=float, default=0.0)
    parser.add_argument('--dynamic_obstacle_ratio', type=float, default=0.3)
    parser.add_argument('--dynamic_obstacle_speed_min', type=float, default=0.2)
    parser.add_argument('--dynamic_obstacle_speed_max', type=float, default=1.0)

    argv = sys.argv[1:]
    # Load the default config file first unless the caller already supplied
    # one, so that later CLI arguments override the file values.
    if not any(a.startswith('@') for a in argv) and os.path.exists(DEFAULT_CONFIG):
        argv = ['@' + DEFAULT_CONFIG] + argv
    args = parser.parse_args(argv)
    if not 0.0 <= args.dynamic_obstacle_scene_prob <= 1.0:
        parser.error('--dynamic_obstacle_scene_prob must be in [0, 1]')
    if not 0.0 <= args.dynamic_obstacle_ratio <= 1.0:
        parser.error('--dynamic_obstacle_ratio must be in [0, 1]')
    if (args.dynamic_obstacle_speed_min < 0.0
            or args.dynamic_obstacle_speed_max
            < args.dynamic_obstacle_speed_min):
        parser.error('invalid dynamic-obstacle speed range')
    if args.protected_zone_radius < 0.0:
        parser.error('--protected_zone_radius must be non-negative')
    if args.obstacle_min_surface_gap < 0.0:
        parser.error('--obstacle_min_surface_gap must be non-negative')
    if args.obstacle_resample_attempts < 1:
        parser.error('--obstacle_resample_attempts must be positive')
    if args.obstacle_scene_restarts < 1:
        parser.error('--obstacle_scene_restarts must be positive')
    if not 0.0 <= args.obstacle_grid_jitter <= 1.0:
        parser.error('--obstacle_grid_jitter must be in [0, 1]')
    if args.obstacle_candidate_multiplier < 1.0:
        parser.error('--obstacle_candidate_multiplier must be at least 1')
    if args.map_size <= 0.0:
        parser.error('--map_size must be positive')
    if min(args.num_cyl, args.num_balls, args.num_vox) < 0:
        parser.error('obstacle counts must be non-negative')
    if args.robot_radius <= 0.0:
        parser.error('--robot_radius must be positive')
    if not 0.0 < args.cyl_radius_min <= args.cyl_radius_max:
        parser.error('invalid cylinder radius range')
    if not 0.0 < args.ball_radius_min <= args.ball_radius_max:
        parser.error('invalid ball radius range')
    if not 0.0 <= args.ball_radius_floor <= args.ball_radius_max:
        parser.error('--ball_radius_floor must be in [0, ball_radius_max]')
    if args.avoid_safe_distance <= args.collision_soft_clearance:
        parser.error(
            '--avoid_safe_distance must exceed --collision_soft_clearance')
    if (args.approach_speed_gain < 0.0
            or args.approach_speed_max < 0.0
            or args.collision_soft_clearance < 0.0):
        parser.error('invalid avoidance-loss parameters')
    if args.speed_distance_gain <= 0.0:
        parser.error('--speed_distance_gain must be positive')
    if not 0.0 <= args.min_desired_speed < args.initial_desired_speed:
        parser.error(
            '--min_desired_speed must be non-negative and below '
            '--initial_desired_speed')
    if args.initial_desired_speed >= args.max_speed:
        parser.error('--initial_desired_speed must be below --max_speed')
    if not 0.0 <= args.speed_min_scale <= 1.0:
        parser.error('--speed_min_scale must be in [0, 1]')
    if (args.speed_clearance_min < 0.0
            or args.speed_clearance_full <= args.speed_clearance_min):
        parser.error(
            '--speed_clearance_full must exceed non-negative '
            '--speed_clearance_min')
    actuator_noise_stds = (
        args.exec_v_scale_std,
        args.exec_w_scale_std,
        args.wheel_bias_std,
    )
    if any(value < 0.0 for value in actuator_noise_stds):
        parser.error('actuator-noise standard deviations must be non-negative')
    if args.dt_noise_std < 0.0:
        parser.error('--dt_noise_std must be non-negative')
    if args.max_action_delay_steps < 0:
        parser.error('--max_action_delay_steps must be non-negative')
    if args.run_name is not None and (
            os.path.basename(args.run_name) != args.run_name
            or args.run_name in ('.', '..')):
        parser.error('--run_name must be one directory name, not a path')
    return args


def configure_run_directories(args, checkpoint=None, timestamp=None):
    """Resolve unique directories for a new run or reuse them on resume."""
    if checkpoint is not None:
        saved_args = checkpoint.get('args') or {}
        run_meta = checkpoint.get('run') or {}
        # The checkpoint location is authoritative even if the run directory
        # was moved after it was created.
        args.save_dir = os.path.dirname(os.path.abspath(args.resume))
        saved_log_dir = run_meta.get('log_dir') or saved_args.get('log_dir')
        if saved_log_dir:
            args.log_dir = saved_log_dir
        elif args.log_dir is None:
            args.log_dir = os.path.join(
                'runs', os.path.basename(args.save_dir))
        args.run_name = (
            run_meta.get('name') or saved_args.get('run_name')
            or os.path.basename(args.save_dir))
        args.save_root = (
            run_meta.get('save_root') or saved_args.get('save_root')
            or os.path.dirname(args.save_dir))
        args.log_root = (
            run_meta.get('log_root') or saved_args.get('log_root')
            or os.path.dirname(args.log_dir))
        return

    save_root = args.save_dir
    log_root = args.log_dir or 'runs'
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    requested_name = args.run_name or f'seed{args.seed}_{timestamp}'
    run_name = requested_name
    suffix = 1
    while (os.path.exists(os.path.join(save_root, run_name))
           or os.path.exists(os.path.join(log_root, run_name))):
        run_name = f'{requested_name}_{suffix:02d}'
        suffix += 1

    args.save_root = save_root
    args.log_root = log_root
    args.run_name = run_name
    args.save_dir = os.path.join(save_root, run_name)
    args.log_dir = os.path.join(log_root, run_name)


def barrier_loss(clearance, approach_weight, safe_distance):
    """Signed-clearance barrier with a detached approach-speed weight."""
    coeff = 10.0
    potential = F.relu(safe_distance - clearance).pow(2)
    return (potential * approach_weight * coeff).mean()


def build_model(args, device):
    return Model(
        dim_obs=6,
        num_waypoints=args.num_waypoints,
        hidden_dim=args.hidden_dim,
        input_w=args.env_width // 2,
        input_h=args.env_height // 2,
        max_forward_step=args.max_forward_step,
        max_lateral_step=args.max_lateral_step,
        max_speed=args.max_speed,
        initial_desired_speed=args.initial_desired_speed,
        min_desired_speed=args.min_desired_speed,
        direct_action=False,
    ).to(device)


def build_mpc(args, device):
    return DifferentiableWaypointMPC(
        num_waypoints=args.num_waypoints,
        horizon=args.mpc_horizon,
        control_lookahead=args.mpc_control_lookahead,
        max_v=args.max_speed,
        max_omega=args.max_omega,
        max_acc_v=args.mpc_max_acc_v,
        max_acc_omega=args.mpc_max_acc_omega,
        max_lateral_acc=args.mpc_max_lateral_acc,
        track_weight=args.mpc_track_weight,
        smooth_weight=args.mpc_smooth_weight,
        initial_velocity_weight=args.mpc_initial_velocity_weight,
        perception_safety_enabled=args.mpc_perception_safety,
        obstacle_safety_clearance=args.mpc_obstacle_clearance,
        obstacle_temperature=args.mpc_obstacle_temperature,
        obstacle_refine_steps=args.mpc_obstacle_refine_steps,
        collect_diagnostics=False,
    ).to(device)


def plot_trajectory(env, p_history, target_pos, i, writer,
                    required_clearance=0.3,
                    waypoint_history=None,
                    mpc_trajectory_history=None,
                    batch_idx=0,
                    tag_suffix='',
                    tag=None,
                    title=None):
    """Plot one batch's executed path with optional waypoint / MPC overlays.

    ``waypoint_history`` / ``mpc_trajectory_history`` are world-frame stacks
    of shape (T, B, N, 2); they are subsampled every 15 steps to avoid a
    cluttered figure.  ``tag`` overrides the default TensorBoard image tag
    (`Trajectory/Batch{batch_idx}{tag_suffix}`) so figures from different
    batches can share one fixed tag and stay browsable via TensorBoard's
    step slider instead of spawning a new tag per batch index.
    """
    if tag is None:
        tag = f'Trajectory/Batch{batch_idx}{tag_suffix}'
    if title is None:
        title = f"iter {i + 1}"
    fig, ax = plt.subplots(figsize=(8, 8))

    if hasattr(env, 'cyl'):
        cylinders = env.cyl[0].detach().cpu().numpy()
        for obs in cylinders:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.4)
            ax.add_artist(circle)
            circle_safe = plt.Circle(
                (obs[0], obs[1]), obs[2] + required_clearance,
                color='red', fill=False, linestyle='--', alpha=0.2)
            ax.add_artist(circle_safe)

    if hasattr(env, 'balls'):
        balls = env.balls[0].detach().cpu().numpy()
        for b in balls:
            circle = plt.Circle((b[0], b[1]), b[3], color='skyblue', alpha=0.4)
            ax.add_artist(circle)

    if hasattr(env, 'voxels'):
        voxels = env.voxels[0].detach().cpu().numpy()
        for v in voxels:
            cx, cy = v[0], v[1]
            rx, ry = v[3], v[4]
            if rx > 5.0 or ry > 5.0:
                continue
            rect = patches.Rectangle(
                (cx - rx, cy - ry), 2 * rx, 2 * ry, color='orange', alpha=0.4)
            ax.add_artist(rect)

    traj = p_history[:, batch_idx, :2].detach().cpu().numpy()
    target = target_pos[batch_idx, :2].detach().cpu().numpy()
    start_pos = traj[0]

    ax.plot(traj[:, 0], traj[:, 1], label='Path', linewidth=2, color='royalblue')
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=8, label='Start')
    ax.scatter(target[0], target[1], c='red', marker='x', s=100,
               label='Target', zorder=10)

    if mpc_trajectory_history is not None:
        # Subsample the MPC's planned horizon paths (green dots).
        mpc_samples = mpc_trajectory_history[::15, batch_idx] \
            .detach().cpu().numpy()
        mpc_points = mpc_samples.reshape(-1, 2)
        ax.scatter(mpc_points[:, 0], mpc_points[:, 1], s=6,
                   color='green', alpha=0.3, label='MPC planned path')

    if waypoint_history is not None:
        # Subsample the policy's predicted waypoints (purple dots).
        waypoint_samples = waypoint_history[::15, batch_idx] \
            .detach().cpu().numpy()
        points = waypoint_samples.reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=10, color='purple',
                   alpha=0.4, label='Predicted waypoints')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title(title)

    all_x = np.concatenate([traj[:, 0], [target[0]], [start_pos[0]]])
    all_y = np.concatenate([traj[:, 1], [target[1]], [start_pos[1]]])

    margin = 2.0
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    span = max(x_max - x_min, y_max - y_min)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    ax.set_xlim(x_mid - span / 2, x_mid + span / 2)
    ax.set_ylim(y_mid - span / 2, y_mid + span / 2)

    writer.add_figure(tag, fig, i + 1)
    plt.close(fig)


def rollout(
    args, env, model, mpc, device, cpu_rng,
    record_visualization=False,
):
    """Run one full differentiable episode and return stacked histories."""
    B = args.batch_size
    h = None

    p_hist, theta_hist, v_hist = [], [], []
    act_hist = []
    # Static scenes can broadcast the final obstacle tensors across time.
    # Dynamic scenes retain detached geometry snapshots so clearance still
    # matches the obstacle position at each control step.
    cyl_hist = [] if env.has_dynamic_obstacles else None
    ball_hist = [] if env.has_dynamic_obstacles else None
    # Global-frame waypoints / MPC trajectories are only materialized on
    # iterations that will actually emit a TensorBoard trajectory figure.
    waypoint_global_hist = [] if record_visualization else None
    mpc_traj_global_hist = [] if record_visualization else None

    # Perceived obstacle state, refreshed at every control step.
    prev_points, prev_valid, prev_clearance = None, None, None

    response_rate_k = torch.rand((B, 2), device=device) \
        * (args.motor_rate_max - args.motor_rate_min) + args.motor_rate_min
    exec_v_scale = None
    if args.exec_v_scale_std > 0.0:
        exec_v_scale = (
            1.0 + torch.randn((B, 1), device=device) * args.exec_v_scale_std
        ).clamp(0.5, 1.5)
    exec_w_scale = None
    if args.exec_w_scale_std > 0.0:
        exec_w_scale = (
            1.0 + torch.randn((B, 1), device=device) * args.exec_w_scale_std
        ).clamp(0.5, 1.5)
    wheel_bias = None
    if args.wheel_bias_std > 0.0:
        wheel_bias = (
            torch.randn((B, 1), device=device) * args.wheel_bias_std)
    # Draw all host-side scalar randomness in two vectorized NumPy calls.
    # This replaces T calls to Python's normalvariate() and avoids a CUDA
    # synchronization formerly used to select the action delay.
    action_delay_steps = int(
        cpu_rng.integers(args.max_action_delay_steps + 1))
    ctl_dts = np.maximum(
        0.01,
        cpu_rng.normal(
            1.0 / args.control_hz,
            args.dt_noise_std,
            size=args.timesteps,
        ),
    ).tolist()

    # Latent motor state for the first-order actuator response.
    motor_actual = torch.zeros((B, 2), device=device)
    if action_delay_steps:
        action_delay_buffer = [
            torch.zeros_like(motor_actual)
            for _ in range(action_delay_steps)
        ]
        action_delay_cursor = 0
    else:
        action_delay_buffer = None

    for ctl_dt in ctl_dts:

        # Exact-state observations (observation noise removed).
        xy_obs = env.p[:, :2]
        yaw_obs = env.theta[:, 2]
        v_obs = env.v[:, 0]
        w_obs = env.omega[:, 2]

        # ---- perception --------------------------------------------------
        # Raw depth stores the ray parameter t; the renderer itself is not
        # differentiated (mpc.py detaches it internally for obstacle points).
        depth, _ = env.render(ctl_dt)
        depth_inv = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
        noise = torch.randn_like(depth_inv) * 0.02
        depth_input = F.max_pool2d(depth_inv + noise, 2, 2)

        # Perceived obstacle inputs only matter when the MPC actually uses
        # them (perception-safety refinement); skip the extraction entirely
        # otherwise to save the per-step perception kernels.
        if args.mpc_perception_safety:
            obstacle_points, obstacle_valid = depth_to_local_obstacle_points(
                depth,
                fov_x_half_tan=args.fov_x_half_tan,
                num_points=args.obstacle_num_points,
                height_fraction=args.obstacle_height_fraction,
                depth_quantile=args.obstacle_depth_quantile,
                min_range=args.obstacle_min_range,
                max_range=args.obstacle_max_range,
            )
            obstacle_velocity = estimate_local_obstacle_velocity(
                obstacle_points, obstacle_valid,
                prev_points, prev_valid,
                v_obs[:, None], w_obs[:, None], ctl_dt,
            )
            emergency_risk, clearance = estimate_emergency_risk(
                depth, prev_clearance, ctl_dt,
                robot_radius=env.drone_radius,
                emergency_distance=args.emergency_distance,
                emergency_ttc=args.emergency_ttc,
            )
            prev_points, prev_valid, prev_clearance = \
                obstacle_points, obstacle_valid, clearance
        else:
            obstacle_points = obstacle_valid = obstacle_velocity = None
            emergency_risk = None

        # ---- goal state in the robot frame (same as train.py) ------------
        vec_global = env.p_target[:, :2] - xy_obs
        cos_th = torch.cos(yaw_obs)
        sin_th = torch.sin(yaw_obs)

        local_x = vec_global[:, 0] * cos_th + vec_global[:, 1] * sin_th
        local_y = vec_global[:, 0] * -sin_th + vec_global[:, 1] * cos_th
        dist_target = torch.sqrt(local_x ** 2 + local_y ** 2)
        scale_mask = dist_target > 10.0
        scale_factor = 10.0 / (dist_target + 1e-6)
        scale = torch.where(scale_mask, scale_factor,
                            torch.ones_like(scale_factor))

        local_x = local_x * scale
        local_y = local_y * scale
        dist_target = dist_target * scale

        state = torch.stack([
            local_x / 10.0,
            local_y / 10.0,
            cos_th,
            sin_th,
            dist_target / 10.0,
            v_obs,
        ], dim=1)

        # ---- waypoint policy --> MPC --> (v, omega) ----------------------
        waypoints, desired_speed, h = model(depth_input, state, h)
        command, trajectory = mpc(
            waypoints,
            desired_speed,
            current_speed=v_obs[:, None],
            dt=ctl_dt,
            current_omega=w_obs[:, None],
            emergency_risk=emergency_risk,
            obstacle_points=obstacle_points,
            obstacle_mask=obstacle_valid,
            obstacle_velocity=obstacle_velocity,
        )

        if record_visualization:
            # Transform body-frame predictions to world coordinates only for
            # the trajectory figures retained by this training script.
            local_wp_x = waypoints[..., 0]
            local_wp_y = waypoints[..., 1]
            true_cos_th = torch.cos(env.theta[:, 2])
            true_sin_th = torch.sin(env.theta[:, 2])
            waypoints_global = torch.stack([
                env.p[:, None, 0] + local_wp_x * true_cos_th[:, None]
                - local_wp_y * true_sin_th[:, None],
                env.p[:, None, 1] + local_wp_x * true_sin_th[:, None]
                + local_wp_y * true_cos_th[:, None],
            ], dim=-1)
            mpc_traj_global = torch.stack([
                env.p[:, None, 0] + trajectory[..., 0] * true_cos_th[:, None]
                - trajectory[..., 1] * true_sin_th[:, None],
                env.p[:, None, 1] + trajectory[..., 0] * true_sin_th[:, None]
                + trajectory[..., 1] * true_cos_th[:, None],
            ], dim=-1)
            waypoint_global_hist.append(waypoints_global.detach())
            mpc_traj_global_hist.append(mpc_traj_global.detach())

        # ---- actuator lag + differentiable CUDA dynamics ------------------
        if action_delay_buffer is None:
            delayed_command = command
        else:
            delayed_command = action_delay_buffer[action_delay_cursor]
            action_delay_buffer[action_delay_cursor] = command
            action_delay_cursor = (
                action_delay_cursor + 1
            ) % action_delay_steps
        motor_v_target = delayed_command[:, 0:1]
        if exec_v_scale is not None:
            motor_v_target = (motor_v_target * exec_v_scale).clamp(
                0.0, args.max_speed)
        motor_w_target = delayed_command[:, 1:2]
        if exec_w_scale is not None:
            motor_w_target = motor_w_target * exec_w_scale
        if wheel_bias is not None:
            motor_w_target = (
                motor_w_target + delayed_command[:, 0:1] * wheel_bias)
        if exec_w_scale is not None or wheel_bias is not None:
            motor_w_target = motor_w_target.clamp(
                -args.max_omega, args.max_omega)
        motor_target = torch.cat(
            [motor_v_target, motor_w_target], dim=1)
        motor_alpha = torch.exp(-response_rate_k * ctl_dt)
        motor_actual = (
            motor_alpha * motor_actual + (1.0 - motor_alpha) * motor_target
        )

        env.run(motor_actual, ctl_dt)

        p_hist.append(env.p.clone())
        theta_hist.append(env.theta.clone())
        v_hist.append(env.v.clone())
        act_hist.append(command)
        if cyl_hist is not None:
            cyl_hist.append(env.cyl)
            ball_hist.append(env.balls)

    p_stack = torch.stack(p_hist)
    cylinders = torch.stack(cyl_hist) if cyl_hist is not None else env.cyl
    balls = torch.stack(ball_hist) if ball_hist is not None else env.balls
    # One batched signed-distance call replaces three geometry kernel groups
    # per control step while retaining the same differentiable loss.
    surface_clearance = env.signed_clearance(
        p_stack, cylinders=cylinders, balls=balls)

    history = {
        'p': p_stack,
        'theta': torch.stack(theta_hist),
        'v': torch.stack(v_hist),
        'act': torch.stack(act_hist),
        'surface_clearance': surface_clearance,
        'ctl_dt': torch.as_tensor(
            ctl_dts, device=device, dtype=env.p.dtype),
    }
    if record_visualization:
        history['waypoints_global'] = torch.stack(waypoint_global_hist)
        history['mpc_trajectory_global'] = torch.stack(mpc_traj_global_hist)
    return history


def compute_losses(args, env, hist):
    """Loss terms mirroring train.py, applied to the MPC command pipeline."""
    AVG_DT = 1.0 / args.control_hz
    p_stack = hist['p']            # (T, B, 3)
    theta_stack = hist['theta']    # (T, B, 3)
    v_stack = hist['v']            # (T, B, 1)
    act_stack = hist['act']        # (T, B, 2)  MPC commands (v, omega)
    surface_clearance = hist['surface_clearance']  # (T, B), signed

    target_expanded = env.p_target.unsqueeze(0).expand(args.timesteps, -1, -1)

    vec_to_target_all = target_expanded[..., :2] - p_stack[..., :2]
    dist_all_steps = torch.norm(vec_to_target_all + 1e-8, dim=-1)
    dist_to_target_vec = dist_all_steps.unsqueeze(-1)
    dir_to_target_all = F.normalize(vec_to_target_all, dim=-1)

    cur_yaw_all = theta_stack[..., 2]
    cur_dir_all = torch.stack(
        [torch.cos(cur_yaw_all), torch.sin(cur_yaw_all)], dim=-1)

    # 1. Pos loss
    arrival_tolerance = 1.0
    loss_pos = F.relu(dist_all_steps - arrival_tolerance).mean()

    # 2. Heading loss
    mask_heading = (dist_all_steps > arrival_tolerance).float()
    raw_heading_loss = 1.0 - F.cosine_similarity(
        cur_dir_all, dir_to_target_all, dim=-1)
    loss_heading = (raw_heading_loss * mask_heading).sum() \
        / (mask_heading.sum() + 1e-5)

    # 3. Velocity loss. Track a per-step target so the policy learns to slow
    # near obstacles and accelerate again in open space. ``surface_clearance``
    # is centre-to-surface distance; subtracting the robot radius converts it
    # to body clearance. Detaching the target prevents the policy from reducing
    # this loss by manipulating its own position or clearance.
    goal_speed = torch.clamp(
        (dist_to_target_vec - arrival_tolerance) * args.speed_distance_gain,
        0.0,
        args.max_speed,
    )
    body_clearance = surface_clearance - args.robot_radius
    clearance_scale = torch.clamp(
        (
            body_clearance - args.speed_clearance_min
        ) / (
            args.speed_clearance_full - args.speed_clearance_min
        ),
        min=args.speed_min_scale,
        max=1.0,
    )
    target_speed_scalar = (
        goal_speed * clearance_scale.unsqueeze(-1)
    ).detach()
    loss_velocity_scalar = F.smooth_l1_loss(
        v_stack, target_speed_scalar)

    # 4. Avoidance loss. The finite-difference approach speed only changes
    # the importance of a frame; detaching it guarantees that the spatial
    # gradient comes exclusively from signed clearance and points outward.
    previous_clearance = torch.cat(
        [surface_clearance[:1], surface_clearance[:-1]], dim=0)
    dt_stack = hist['ctl_dt'].reshape(-1, 1).clamp_min(1e-3)
    approach_speed = (
        -(surface_clearance - previous_clearance) / dt_stack
    ).clamp(min=0.0, max=args.approach_speed_max)
    approach_weight = (
        1.0 + args.approach_speed_gain * approach_speed
    ).detach()
    loss_avoid = barrier_loss(
        surface_clearance, approach_weight, args.avoid_safe_distance)

    # 5. Collision loss
    dist_diff = args.collision_soft_clearance - surface_clearance
    loss_collide = F.softplus(dist_diff * 10.0).mean() * 2.0

    # 6. Smooth loss (on executed MPC commands)
    loss_smooth = (act_stack.diff(1, 0) / AVG_DT).pow(2).mean()

    # 7. Bias loss (keep velocity vector aligned with the target direction)
    v_real_vec = torch.stack([
        v_stack[..., 0] * torch.cos(theta_stack[..., 2]),
        v_stack[..., 0] * torch.sin(theta_stack[..., 2])
    ], dim=-1)
    v_proj_val = (v_real_vec * dir_to_target_all).sum(dim=-1, keepdim=True)
    v_proj_vec = v_proj_val * dir_to_target_all
    loss_bias = F.mse_loss(v_real_vec, v_proj_vec)

    # 8. Energy loss
    loss_action_energy = act_stack.pow(2).mean()

    # 9. Success / collision metrics (logged, not part of the loss)
    # Success: entered the 1 m goal radius at any timestep AND no collision
    # over the whole episode. Physical contact occurs when centre-to-surface
    # signed clearance is no greater than the robot radius.
    with torch.no_grad():
        reached_goal = (dist_all_steps <= arrival_tolerance).any(dim=0)   # (B,)
        collided = (surface_clearance <= env.drone_radius).any(dim=0)    # (B,)
        success_rate = (reached_goal & ~collided).float().mean()
        collision_rate = collided.float().mean()

    total_loss = args.coef_pos * loss_pos \
        + args.coef_heading * loss_heading \
        + args.coef_v * loss_velocity_scalar \
        + args.coef_obj_avoidance * loss_avoid \
        + args.coef_collide * loss_collide \
        + args.coef_smooth * loss_smooth \
        + args.coef_bias * loss_bias \
        + args.coef_energy * loss_action_energy

    parts = {
        'Loss/Total': total_loss,
        'Loss/Pos': loss_pos,
        'Loss/Heading': loss_heading,
        'Loss/Velocity': loss_velocity_scalar,
        'Loss/Avoid': loss_avoid,
        'Loss/Collide': loss_collide,
        'Loss/Smooth': loss_smooth,
        'Loss/Bias': loss_bias,
        'Loss/Energy': loss_action_energy,
        'Metric/FinalDist': dist_all_steps[-1].mean(),
        'Metric/MeanSpeed': v_stack.mean(),
        'Metric/SuccessRate': success_rate,
        'Metric/CollisionRate': collision_rate,
    }
    return total_loss, parts


def save_checkpoint(args, model, optim, sched, i):
    ckpt = {
        'iter': i + 1,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'sched': sched.state_dict(),
        'args': vars(args).copy(),
        'run': {
            'name': args.run_name,
            'save_root': args.save_root,
            'log_root': args.log_root,
            'save_dir': args.save_dir,
            'log_dir': args.log_dir,
        },
    }
    os.makedirs(args.save_dir, exist_ok=True)
    numbered = os.path.join(
        args.save_dir, f'checkpoint_mpc_{i + 1}.pth')
    latest = os.path.join(args.save_dir, 'latest.pth')
    torch.save(ckpt, numbered)
    torch.save(ckpt, latest)
    return numbered


def main():
    args = parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
    cpu_rng = np.random.default_rng(args.seed if args.seed else None)

    device = torch.device('cuda')
    ckpt = None
    start_iter = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        start_iter = ckpt.get('iter', 0)

    configure_run_directories(args, checkpoint=ckpt)
    os.makedirs(args.save_dir, exist_ok=ckpt is not None)
    writer = SummaryWriter(
        log_dir=args.log_dir,
        purge_step=(start_iter + 1 if ckpt is not None else None),
    )
    print(f"Run name: {args.run_name}")
    print(f"Checkpoint directory: {args.save_dir}")
    print(f"TensorBoard directory: {args.log_dir}")

    env = Env(args.batch_size, args.env_width, args.env_height,
              args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan,
              ground_voxels=True,
              diff_nearest_pt=args.diff_nearest_pt,
              map_size=args.map_size,
              num_cyl=args.num_cyl,
              num_balls=args.num_balls,
              num_vox=args.num_vox,
              robot_radius=args.robot_radius,
              cyl_radius_min=args.cyl_radius_min,
              cyl_radius_max=args.cyl_radius_max,
              ball_radius_min=args.ball_radius_min,
              ball_radius_max=args.ball_radius_max,
              ball_radius_floor=args.ball_radius_floor,
              randomize_start_goal=args.randomize_start_goal,
              protected_zone_radius=args.protected_zone_radius,
              obstacle_min_surface_gap=args.obstacle_min_surface_gap,
              obstacle_resample_attempts=args.obstacle_resample_attempts,
              obstacle_scene_restarts=args.obstacle_scene_restarts,
              obstacle_layout=args.obstacle_layout,
              obstacle_grid_jitter=args.obstacle_grid_jitter,
              obstacle_candidate_multiplier=(
                  args.obstacle_candidate_multiplier),
              dynamic_obstacle_scene_prob=(
                  args.dynamic_obstacle_scene_prob),
              dynamic_obstacle_ratio=args.dynamic_obstacle_ratio,
              dynamic_obstacle_speed_min=args.dynamic_obstacle_speed_min,
              dynamic_obstacle_speed_max=args.dynamic_obstacle_speed_max,
              dynamic_obstacle_seed=torch.initial_seed() + 104729)

    model = build_model(args, device)
    mpc = build_mpc(args, device)

    optim = AdamW(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        if 'optim' in ckpt:
            optim.load_state_dict(ckpt['optim'])
        if 'sched' in ckpt:
            sched.load_state_dict(ckpt['sched'])
        print(f"Resumed from iteration {start_iter}")

    metric_names = (
        'Loss/Total',
        'Loss/Pos',
        'Loss/Heading',
        'Loss/Velocity',
        'Loss/Avoid',
        'Loss/Collide',
        'Loss/Smooth',
        'Loss/Bias',
        'Loss/Energy',
        'Metric/FinalDist',
        'Metric/MeanSpeed',
        'Metric/SuccessRate',
        'Metric/CollisionRate',
    )
    scalar_sums = torch.zeros(len(metric_names), device=device)
    scalar_count = 0

    pbar = tqdm(range(start_iter, args.num_iters), ncols=100,
                initial=start_iter, total=args.num_iters)

    for i in pbar:
        env.reset()

        record_visualization = (i + 1) % args.plot_interval == 0
        hist = rollout(
            args, env, model, mpc, device, cpu_rng,
            record_visualization=record_visualization)
        total_loss, parts = compute_losses(args, env, hist)

        # Device-side assertion retains non-finite protection without forcing
        # a GPU->CPU synchronization on every training iteration.
        torch._assert_async(
            torch.isfinite(total_loss),
            f'Non-finite loss detected at iteration {i}',
        )

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        with torch.no_grad():
            scalar_sums.add_(torch.stack([
                parts[name].detach() for name in metric_names
            ]))
            scalar_count += 1
            if (i + 1) % args.log_interval == 0:
                averages = (scalar_sums / scalar_count).cpu().tolist()
                logged = dict(zip(metric_names, averages))
                for name, average in logged.items():
                    writer.add_scalar(name, average, i + 1)
                scalar_sums.zero_()
                scalar_count = 0
                pbar.set_description(
                    f"L:{logged['Loss/Total']:.2f}|"
                    f"SR:{logged['Metric/SuccessRate'] * 100:.1f}%|"
                    f"CR:{logged['Metric/CollisionRate'] * 100:.1f}%|"
                    f"D:{logged['Metric/FinalDist']:.2f}")

            if (i + 1) % args.plot_interval == 0:
                # Regular batch-0 figure with predicted waypoints overlaid.
                plot_trajectory(
                    env, hist['p'], env.p_target, i, writer,
                    waypoint_history=hist['waypoints_global'],
                    batch_idx=0)

                # Worst-batch figure: physical collisions dominate the score,
                # then final distance decides among equally safe trajectories.
                min_clearance = hist['surface_clearance'].min(dim=0).values
                final_dist = (env.p_target[:, :2]
                              - hist['p'][-1, :, :2]).norm(dim=-1)
                collided_flag = (min_clearance <= env.drone_radius).float()
                worst_score = collided_flag * 1e6 + final_dist
                worst_idx = int(worst_score.argmax().item())
                # Fixed tag so every plot interval appends to the same
                # TensorBoard entry (step slider) regardless of which batch
                # was worst; the batch index is kept in the figure title.
                plot_trajectory(
                    env, hist['p'], env.p_target, i, writer,
                    waypoint_history=hist['waypoints_global'],
                    mpc_trajectory_history=hist['mpc_trajectory_global'],
                    batch_idx=worst_idx, tag='Trajectory/Worst',
                    title=f"iter {i + 1} | worst batch {worst_idx}")

            if (i + 1) % args.ckpt_interval == 0:
                path = save_checkpoint(args, model, optim, sched, i)
                tqdm.write(f"Saved checkpoint: {path}")

    save_checkpoint(args, model, optim, sched, args.num_iters - 1)
    writer.close()


if __name__ == '__main__':
    main()
