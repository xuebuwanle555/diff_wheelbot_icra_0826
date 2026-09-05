"""Train a recurrent direct-action PPO baseline in the icra CUDA environment.

The baseline uses the same depth preprocessing, six-dimensional goal state,
domain randomization, actuator lag and start/goal distribution as
``train_mpc.py``.  Unlike the proposed method, it predicts linear and angular
velocity directly and does not use waypoints, differentiable dynamics losses,
or MPC.

Usage:
    python3 train_ppo.py @configs/ppo_train.args
"""

import argparse
import math
import os
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env_cuda import Env
from ppo_model import PPOActorCritic
from train_mpc import (
    obstacle_curriculum_probabilities,
    obstacle_curriculum_state,
)


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
        description='Diff-Wheelbot direct-action PPO baseline training',
    )
    parser.add_argument('--resume', default=None)
    parser.add_argument('--save_dir', default='save/ppo_baseline')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of parallel CUDA environments')
    parser.add_argument('--rollout_steps', type=int, default=150)
    parser.add_argument('--randomize_horizon', type=str2bool, default=True,
                        help='Sample the rollout horizon per update, matching '
                             'train_mpc.py so the recurrent policy cannot '
                             'use the step count as a progress cue')
    parser.add_argument('--min_timesteps', type=int, default=135)
    parser.add_argument('--max_timesteps', type=int, default=165)
    parser.add_argument('--total_timesteps', type=int, default=20_000_000)
    parser.add_argument('--save_every', type=int, default=50,
                        help='Checkpoint interval in PPO updates')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_coef', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--minibatch_envs', type=int, default=32,
                        help='Whole recurrent trajectories per minibatch')
    parser.add_argument('--target_kl', type=float, default=0.03)
    parser.add_argument('--initial_log_std', type=float, default=-0.7)
    parser.add_argument('--anneal_lr', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=192,
                        help='Shared CNN/GRU capacity; 192 matches main model')

    # Actuation randomization, matching train_mpc.py argument names.
    parser.add_argument('--control_hz', type=float, default=15.0)
    parser.add_argument('--dt_noise_std', type=float, default=0.005)
    parser.add_argument('--min_ctl_dt', type=float, default=0.01)
    parser.add_argument('--motor_rate_min', type=float, default=3.0)
    parser.add_argument('--motor_rate_max', type=float, default=8.0)
    parser.add_argument('--max_action_delay_steps', type=int, default=1)
    parser.add_argument('--exec_v_scale_std', type=float, default=0.0)
    parser.add_argument('--exec_w_scale_std', type=float, default=0.0)
    parser.add_argument('--wheel_bias_std', type=float, default=0.0)

    # Observation preprocessing, matching train_mpc.py.
    parser.add_argument('--depth_noise_std', type=float, default=0.02)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
    parser.add_argument('--grad_decay', type=float, default=0.6)
    parser.add_argument('--env_width', type=int, default=64)
    parser.add_argument('--env_height', type=int, default=48)

    # Velocity limits and goal semantics shared with the main model.
    parser.add_argument('--max_speed', type=float, default=2.0)
    parser.add_argument('--max_omega', type=float, default=3.0)
    parser.add_argument('--cruise_v', type=float, default=1.5,
                        help='Initial policy speed bias; must be in '
                             '(0, max_speed)')
    parser.add_argument('--goal_stop_distance', type=float, default=0.5)
    parser.add_argument('--goal_state_max_distance', type=float, default=6.0,
                        help='Clamp/scale radius for goal coordinates in the '
                             'state vector; must equal the main model value')
    # Clearance is robot-surface to obstacle-surface distance.
    parser.add_argument('--safety_margin', type=float, default=0.25)

    # PPO reward terms.
    parser.add_argument('--reward_progress', type=float, default=2.0)
    parser.add_argument('--reward_arrival', type=float, default=25.0)
    parser.add_argument('--reward_collision', type=float, default=-25.0)
    parser.add_argument('--reward_clearance', type=float, default=0.5)
    parser.add_argument('--reward_time', type=float, default=-0.01)
    parser.add_argument('--reward_smooth', type=float, default=0.05)
    parser.add_argument('--reward_omega', type=float, default=0.01)

    # Environment settings, mirroring train_mpc.py / config files.
    parser.add_argument('--map_size', type=float, default=14.0)
    parser.add_argument('--num_cyl', type=int, default=18)
    parser.add_argument('--num_balls', type=int, default=10)
    parser.add_argument('--num_vox', type=int, default=8)
    parser.add_argument('--robot_radius', type=float, default=0.24)
    parser.add_argument('--cyl_radius_min', type=float, default=0.15)
    parser.add_argument('--cyl_radius_max', type=float, default=0.45)
    parser.add_argument('--cyl_height_min', type=float, default=0.5)
    parser.add_argument('--cyl_height_max', type=float, default=1.5)
    parser.add_argument('--ball_radius_min', type=float, default=0.15)
    parser.add_argument('--ball_radius_max', type=float, default=0.45)
    parser.add_argument('--ball_radius_floor', type=float, default=0.25)
    parser.add_argument('--initial_yaw_noise', type=float, default=0.26)
    parser.add_argument('--randomize_start_goal', type=str2bool, default=True)
    parser.add_argument('--protected_zone_radius', type=float, default=1.0)
    parser.add_argument(
        '--obstacle_layout', choices=('nonoverlap', 'stratified'),
        default='stratified')
    parser.add_argument('--obstacle_grid_jitter', type=float, default=0.35)
    parser.add_argument(
        '--obstacle_candidate_multiplier', type=float, default=2.0)
    parser.add_argument('--dynamic_obstacle_scene_prob', type=float, default=0.0)
    parser.add_argument('--dynamic_obstacle_ratio', type=float, default=0.0)
    parser.add_argument('--dynamic_obstacle_speed_min', type=float, default=0.0)
    parser.add_argument('--dynamic_obstacle_speed_max', type=float, default=0.0)

    # Obstacle-density curriculum, mirroring train_mpc.py.  Boundaries are
    # expressed in PPO updates (train_mpc uses iterations); pick them at the
    # same fractions of the total update budget.
    parser.add_argument('--obstacle_curriculum', type=str2bool, default=False)
    parser.add_argument('--obstacle_curriculum_mode',
                        choices=('staged', 'mixed'), default='mixed')
    parser.add_argument('--obstacle_curriculum_boundaries', type=int, nargs=3,
                        default=[3000, 6000, 9000])
    parser.add_argument('--obstacle_curriculum_num_cyl', type=int, nargs=4,
                        default=[20, 24, 27, 27])
    parser.add_argument('--obstacle_curriculum_num_balls', type=int, nargs=4,
                        default=[11, 13, 15, 15])
    parser.add_argument('--obstacle_curriculum_num_vox', type=int, nargs=4,
                        default=[8, 9, 10, 10])
    parser.add_argument('--obstacle_curriculum_mix_boundaries', type=int,
                        nargs=2, default=[3000, 9000])
    parser.add_argument('--obstacle_curriculum_phase2_probs', type=float,
                        nargs=3, default=[0.70, 0.30, 0.00])
    parser.add_argument('--obstacle_curriculum_phase3_probs', type=float,
                        nargs=3, default=[0.50, 0.40, 0.10])

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive = (
        'batch_size', 'rollout_steps', 'total_timesteps', 'save_every',
        'hidden_dim',
        'lr', 'gamma', 'gae_lambda', 'clip_coef', 'max_grad_norm',
        'update_epochs', 'minibatch_envs', 'control_hz', 'min_ctl_dt',
        'max_speed', 'max_omega', 'motor_rate_min', 'motor_rate_max',
        'map_size', 'robot_radius',
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.minibatch_envs > args.batch_size:
        parser.error('--minibatch_envs cannot exceed --batch_size')
    if not 0.0 < args.cruise_v < args.max_speed:
        parser.error('Require 0 < --cruise_v < --max_speed')
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        parser.error('--gamma and --gae_lambda must be in (0, 1]')
    if args.max_action_delay_steps < 0:
        parser.error('--max_action_delay_steps must be non-negative')
    if args.motor_rate_max < args.motor_rate_min:
        parser.error('Invalid motor-rate range')
    if args.goal_stop_distance < 0.0 or args.safety_margin <= 0.0:
        parser.error(
            'Goal-stop distance must be non-negative and safety margin '
            'positive')
    if not 0.0 <= args.dynamic_obstacle_scene_prob <= 1.0:
        parser.error('--dynamic_obstacle_scene_prob must be in [0, 1]')
    if not 0.0 <= args.dynamic_obstacle_ratio <= 1.0:
        parser.error('--dynamic_obstacle_ratio must be in [0, 1]')
    if (args.dynamic_obstacle_speed_min < 0.0
            or args.dynamic_obstacle_speed_max
            < args.dynamic_obstacle_speed_min):
        parser.error('Invalid dynamic-obstacle speed range')
    noise_stds = (
        args.depth_noise_std, args.dt_noise_std, args.exec_v_scale_std,
        args.exec_w_scale_std, args.wheel_bias_std, args.initial_yaw_noise,
    )
    if any(value < 0.0 for value in noise_stds):
        parser.error('Noise standard deviations must be non-negative')
    if min(args.num_cyl, args.num_balls, args.num_vox) < 0:
        parser.error('Obstacle counts must be non-negative')
    if not 0.0 < args.cyl_radius_min <= args.cyl_radius_max:
        parser.error('Invalid cylinder radius range')
    if not 0.0 < args.ball_radius_min <= args.ball_radius_max:
        parser.error('Invalid ball radius range')
    if not 0.0 <= args.ball_radius_floor <= args.ball_radius_max:
        parser.error('--ball_radius_floor must be in [0, ball_radius_max]')
    if args.protected_zone_radius < 0.0:
        parser.error('--protected_zone_radius must be non-negative')
    if not 0.0 <= args.obstacle_grid_jitter <= 1.0:
        parser.error('--obstacle_grid_jitter must be in [0, 1]')
    if args.obstacle_candidate_multiplier < 1.0:
        parser.error('--obstacle_candidate_multiplier must be at least 1')
    if not 0.0 < args.cyl_height_min <= args.cyl_height_max:
        parser.error('Invalid cylinder height range')
    if args.goal_state_max_distance <= 0.0:
        parser.error('--goal_state_max_distance must be positive')
    if args.randomize_horizon:
        if args.min_timesteps <= 0 or args.max_timesteps < args.min_timesteps:
            parser.error('--min_timesteps must be positive and '
                         '<= --max_timesteps')
    curriculum_counts = (
        args.obstacle_curriculum_num_cyl,
        args.obstacle_curriculum_num_balls,
        args.obstacle_curriculum_num_vox,
    )
    if any(min(counts) < 0 for counts in curriculum_counts):
        parser.error('obstacle curriculum counts must be non-negative')
    if any(
            later < earlier
            for counts in curriculum_counts
            for earlier, later in zip(counts, counts[1:])):
        parser.error('obstacle curriculum counts must be non-decreasing')
    boundaries = args.obstacle_curriculum_boundaries
    if any(boundary <= 0 for boundary in boundaries) or any(
            later <= earlier
            for earlier, later in zip(boundaries, boundaries[1:])):
        parser.error(
            '--obstacle_curriculum_boundaries must be positive and strictly '
            'increasing')
    mix_boundaries = args.obstacle_curriculum_mix_boundaries
    if (mix_boundaries[0] <= 0
            or mix_boundaries[1] <= mix_boundaries[0]):
        parser.error(
            '--obstacle_curriculum_mix_boundaries must be positive and '
            'strictly increasing')
    if args.obstacle_curriculum:
        total_updates = math.ceil(
            args.total_timesteps / (args.batch_size * args.rollout_steps))
        if (args.obstacle_curriculum_mode == 'staged'
                and boundaries[-1] >= total_updates):
            parser.error(
                'the final obstacle curriculum stage must begin before the '
                'total number of PPO updates')
        if (args.obstacle_curriculum_mode == 'mixed'
                and mix_boundaries[-1] >= total_updates):
            parser.error(
                'the final mixed curriculum phase must begin before the '
                'total number of PPO updates')
    for name, probabilities in (
            ('--obstacle_curriculum_phase2_probs',
             args.obstacle_curriculum_phase2_probs),
            ('--obstacle_curriculum_phase3_probs',
             args.obstacle_curriculum_phase3_probs)):
        if any(probability < 0.0 for probability in probabilities):
            parser.error(f'{name} values must be non-negative')
        if abs(sum(probabilities) - 1.0) > 1e-6:
            parser.error(f'{name} values must sum to 1')


def plot_trajectory(
    env, p_history, target_pos, step, writer,
    required_clearance=0.3, batch_idx=0, tag_suffix='',
):
    fig, ax = plt.subplots(figsize=(8, 8))

    if hasattr(env, 'cyl') and env.cyl.shape[1] > 0:
        cylinders = env.cyl[batch_idx].detach().cpu().numpy()
        for obs in cylinders:
            circle = plt.Circle(
                (obs[0], obs[1]), obs[2], color='gray', alpha=0.4)
            ax.add_artist(circle)
            circle_safe = plt.Circle(
                (obs[0], obs[1]),
                obs[2] + env.drone_radius + required_clearance,
                color='red', fill=False, linestyle='--', alpha=0.2,
            )
            ax.add_artist(circle_safe)

    if hasattr(env, 'balls') and env.balls.shape[1] > 0:
        balls = env.balls[batch_idx].detach().cpu().numpy()
        for b in balls:
            circle = plt.Circle(
                (b[0], b[1]), b[3], color='skyblue', alpha=0.4)
            ax.add_artist(circle)

    if hasattr(env, 'voxels') and env.voxels.shape[1] > 0:
        voxels = env.voxels[batch_idx].detach().cpu().numpy()
        for v in voxels:
            cx, cy = v[0], v[1]
            rx, ry = v[3], v[4]
            if rx > 5.0 or ry > 5.0:
                continue
            rect = patches.Rectangle(
                (cx - rx, cy - ry), 2 * rx, 2 * ry,
                color='orange', alpha=0.4,
            )
            ax.add_artist(rect)

    traj = p_history[:, batch_idx, :2].detach().cpu().numpy()
    target = target_pos[batch_idx, :2].detach().cpu().numpy()
    start_pos = traj[0]

    ax.plot(
        traj[:, 0], traj[:, 1], label='Path', linewidth=2, color='royalblue')
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=8, label='Start')
    ax.scatter(
        target[0], target[1], c='red', marker='x', s=100, label='Target',
        zorder=10)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title(f'step {step}  (batch {batch_idx})')

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

    writer.add_figure(f'Trajectory/WorstBatch{tag_suffix}', fig, step)
    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_state(env, yaw_obs, v_obs, args):
    """Goal state in the robot frame, exactly as in train_mpc.py."""
    vec_global = env.p_target[:, :2] - env.p[:, :2]
    cos_th = torch.cos(yaw_obs)
    sin_th = torch.sin(yaw_obs)
    local_x = vec_global[:, 0] * cos_th + vec_global[:, 1] * sin_th
    local_y = vec_global[:, 0] * -sin_th + vec_global[:, 1] * cos_th
    dist_target = torch.sqrt(local_x ** 2 + local_y ** 2)
    max_distance = args.goal_state_max_distance
    scale = torch.where(
        dist_target > max_distance,
        max_distance / (dist_target + 1e-6),
        torch.ones_like(dist_target),
    )
    # Speed is normalised by the configured cap so the state is
    # invariant to max_speed, identical to train_mpc.py.
    return torch.stack([
        local_x * scale / max_distance,
        local_y * scale / max_distance,
        cos_th,
        sin_th,
        dist_target * scale / max_distance,
        v_obs / args.max_speed,
    ], dim=1)


def build_env(args, device):
    return Env(
        args.batch_size, args.env_width, args.env_height,
        args.grad_decay, device,
        fov_x_half_tan=args.fov_x_half_tan,
        ground_voxels=True,
        map_size=args.map_size,
        num_cyl=args.num_cyl,
        num_balls=args.num_balls,
        num_vox=args.num_vox,
        robot_radius=args.robot_radius,
        cyl_radius_min=args.cyl_radius_min,
        cyl_radius_max=args.cyl_radius_max,
        cyl_height_min=args.cyl_height_min,
        cyl_height_max=args.cyl_height_max,
        ball_radius_min=args.ball_radius_min,
        ball_radius_max=args.ball_radius_max,
        ball_radius_floor=args.ball_radius_floor,
        initial_yaw_noise=args.initial_yaw_noise,
        randomize_start_goal=args.randomize_start_goal,
        protected_zone_radius=args.protected_zone_radius,
        obstacle_layout=args.obstacle_layout,
        obstacle_grid_jitter=args.obstacle_grid_jitter,
        obstacle_candidate_multiplier=args.obstacle_candidate_multiplier,
        dynamic_obstacle_scene_prob=args.dynamic_obstacle_scene_prob,
        dynamic_obstacle_ratio=args.dynamic_obstacle_ratio,
        dynamic_obstacle_speed_min=args.dynamic_obstacle_speed_min,
        dynamic_obstacle_speed_max=args.dynamic_obstacle_speed_max,
        dynamic_obstacle_seed=torch.initial_seed() + 104729,
    )


@torch.no_grad()
def collect_rollout(args, env, model, cpu_rng, rollout_steps):
    device = env.p.device
    batch_size = args.batch_size
    env.reset()
    hidden = None
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    arrived = torch.zeros_like(active)
    collided = torch.zeros_like(active)
    clearance_violated = torch.zeros_like(active)
    episode_return = torch.zeros(batch_size, device=device)

    motor_actual = torch.zeros((batch_size, 2), device=device)
    previous_command = torch.zeros_like(motor_actual)
    response_rate = torch.rand((batch_size, 2), device=device)
    response_rate = response_rate * (
        args.motor_rate_max - args.motor_rate_min
    ) + args.motor_rate_min
    exec_v_scale = None
    if args.exec_v_scale_std > 0.0:
        exec_v_scale = (
            1.0 + torch.randn((batch_size, 1), device=device)
            * args.exec_v_scale_std
        ).clamp(0.5, 1.5)
    exec_w_scale = None
    if args.exec_w_scale_std > 0.0:
        exec_w_scale = (
            1.0 + torch.randn((batch_size, 1), device=device)
            * args.exec_w_scale_std
        ).clamp(0.5, 1.5)
    wheel_bias = None
    if args.wheel_bias_std > 0.0:
        wheel_bias = torch.randn(
            (batch_size, 1), device=device) * args.wheel_bias_std
    action_delay_steps = int(
        cpu_rng.integers(args.max_action_delay_steps + 1))
    if action_delay_steps:
        action_delay_buffer = [
            torch.zeros_like(motor_actual)
            for _ in range(action_delay_steps)
        ]
        action_delay_cursor = 0
    else:
        action_delay_buffer = None

    ctl_dts = np.maximum(
        args.min_ctl_dt,
        cpu_rng.normal(
            1.0 / args.control_hz, args.dt_noise_std,
            size=rollout_steps,
        ),
    ).tolist()

    initial_distance = torch.linalg.vector_norm(
        env.p_target[:, :2] - env.p[:, :2], dim=1)
    previous_distance = initial_distance
    terminal_distance = initial_distance.clone()
    min_clearance = torch.full((batch_size,), float('inf'), device=device)

    depths = []
    states = []
    latents = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    valid_steps = []
    command_speeds = []
    p_hist_list = [env.p.clone()]

    for ctl_dt in ctl_dts:
        valid = active.clone()
        # Exact-state observations, identical to train_mpc.py.
        yaw_obs = env.theta[:, 2]
        v_obs = env.v[:, 0]

        depth, _ = env.render(ctl_dt)
        depth_inv = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
        if args.depth_noise_std > 0.0:
            depth_inv = depth_inv + (
                torch.randn_like(depth_inv) * args.depth_noise_std)
        depth_input = F.max_pool2d(depth_inv, 2, 2)

        state = make_state(env, yaw_obs, v_obs, args)

        command, latent, log_prob, value, hidden = model.sample_action(
            depth_input, state, hidden)
        command = torch.where(
            valid[:, None], command, torch.zeros_like(command))

        # Actuator lag, identical to train_mpc.rollout.
        if action_delay_buffer is None:
            delayed_command = command
        else:
            delayed_command = action_delay_buffer[action_delay_cursor]
            action_delay_buffer[action_delay_cursor] = command
            action_delay_cursor = (
                action_delay_cursor + 1) % action_delay_steps
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
        motor_target = torch.cat([motor_v_target, motor_w_target], dim=1)
        motor_alpha = torch.exp(-response_rate * ctl_dt)
        motor_actual = (
            motor_alpha * motor_actual + (1.0 - motor_alpha) * motor_target)
        env.run(motor_actual, ctl_dt)

        distance = torch.linalg.vector_norm(
            env.p_target[:, :2] - env.p[:, :2], dim=1)
        clearance = env.signed_clearance(subtract_robot_radius=True)
        step_arrival = valid & (distance <= args.goal_stop_distance)
        step_collision = valid & (clearance <= 0.0)
        done = step_arrival | step_collision

        progress_reward = args.reward_progress * (
            previous_distance - distance)
        clearance_fraction = (
            F.relu(args.safety_margin - clearance) / args.safety_margin
        ).clamp(max=2.0)
        delta_command = (command - previous_command) / torch.tensor(
            [args.max_speed, args.max_omega], device=device)
        smooth_cost = delta_command.square().sum(dim=1)
        omega_cost = (command[:, 1] / args.max_omega).square()
        reward = (
            progress_reward
            + args.reward_time
            - args.reward_clearance * clearance_fraction
            - args.reward_smooth * smooth_cost
            - args.reward_omega * omega_cost
            + args.reward_arrival * step_arrival.float()
            + args.reward_collision * step_collision.float()
        )
        reward = torch.where(valid, reward, torch.zeros_like(reward))

        depths.append(depth_input)
        states.append(state)
        latents.append(latent)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        valid_steps.append(valid)
        command_speeds.append(command[:, 0])

        episode_return += reward
        arrived |= step_arrival
        collided |= step_collision
        clearance_violated |= valid & (clearance <= args.safety_margin)
        min_clearance = torch.where(
            valid, torch.minimum(min_clearance, clearance), min_clearance)
        terminal_distance = torch.where(valid, distance, terminal_distance)
        previous_distance = torch.where(valid, distance, previous_distance)
        previous_command = command
        active &= ~done
        p_hist_list.append(env.p.clone())

    rollout = {
        'depth': torch.stack(depths),
        'state': torch.stack(states),
        'latent': torch.stack(latents),
        'old_log_prob': torch.stack(log_probs),
        'old_value': torch.stack(values),
        'reward': torch.stack(rewards),
        'done': torch.stack(dones),
        'valid': torch.stack(valid_steps),
    }
    # A rollout horizon is a time-limit truncation, not an MDP terminal.  For
    # environments that are still active, bootstrap V(s_T) from the final
    # observation.  Treating it as zero teaches an artificial end-of-horizon
    # value drop and can make a recurrent policy slow down near T.
    if bool(active.any()):
        yaw_obs = env.theta[:, 2]
        v_obs = env.v[:, 0]
        depth, _ = env.render(ctl_dts[-1])
        depth_inv = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
        if args.depth_noise_std > 0.0:
            depth_inv = depth_inv + (
                torch.randn_like(depth_inv) * args.depth_noise_std)
        final_depth = F.max_pool2d(depth_inv, 2, 2)
        final_state = make_state(env, yaw_obs, v_obs, args)
        _, bootstrap_value, _ = model(final_depth, final_state, hidden)
        bootstrap_value = torch.where(
            active, bootstrap_value, torch.zeros_like(bootstrap_value))
    else:
        bootstrap_value = torch.zeros(batch_size, device=device)
    rollout['bootstrap_value'] = bootstrap_value
    rollout['advantage'], rollout['return'] = compute_gae(args, rollout)
    rollout['p_history'] = torch.stack(p_hist_list)  # (T+1, B, 3)
    rollout['min_clearance'] = min_clearance
    rollout['terminal_distance'] = terminal_distance
    metrics = {
        'return': episode_return.mean().item(),
        'arrival_rate': arrived.float().mean().item(),
        'collision_rate': collided.float().mean().item(),
        'success_rate': (arrived & ~collided).float().mean().item(),
        'clearance_safe_arrival_rate':
            (arrived & ~clearance_violated).float().mean().item(),
        'clearance_violation_rate':
            clearance_violated.float().mean().item(),
        'final_distance': terminal_distance.mean().item(),
        'min_clearance': min_clearance[
            torch.isfinite(min_clearance)].mean().item(),
        'mean_speed_command': torch.stack(command_speeds)[
            rollout['valid']].mean().item(),
        'valid_transitions': rollout['valid'].sum().item(),
        'dynamic_scene_rate':
            env.dynamic_scene_mask.float().mean().item(),
    }
    if env.has_dynamic_obstacles:
        metrics['moving_obstacle_rate'] = (
            (
                (env.cyl_velocity.norm(dim=-1) > 0.0).sum()
                + (env.ball_velocity.norm(dim=-1) > 0.0).sum()
            ).float()
            / max(batch_size * (env.cyl.shape[1] + env.balls.shape[1]), 1)
        ).item()
    return rollout, metrics


def compute_gae(args, rollout):
    rewards = rollout['reward']
    values = rollout['old_value']
    dones = rollout['done']
    valid = rollout['valid']
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    last_value = rollout.get(
        'bootstrap_value', torch.zeros_like(last_advantage))

    for step in reversed(range(horizon)):
        next_value = (
            last_value if step == horizon - 1 else values[step + 1])
        nonterminal = (~dones[step]).float()
        delta = (
            rewards[step] + args.gamma * next_value * nonterminal
            - values[step])
        last_advantage = (
            delta
            + args.gamma * args.gae_lambda * nonterminal * last_advantage)
        last_advantage = torch.where(
            valid[step], last_advantage, torch.zeros_like(last_advantage))
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def evaluate_sequence(model, rollout, env_indices):
    hidden = None
    log_probs = []
    entropies = []
    values = []
    for step in range(rollout['depth'].shape[0]):
        log_prob, entropy, value, hidden = model.evaluate_latent(
            rollout['depth'][step, env_indices],
            rollout['state'][step, env_indices],
            rollout['latent'][step, env_indices],
            hidden,
        )
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
    return (
        torch.stack(log_probs), torch.stack(entropies), torch.stack(values))


def ppo_update(args, model, optimizer, rollout):
    valid_advantages = rollout['advantage'][rollout['valid']]
    advantage_mean = valid_advantages.mean()
    advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
    normalized_advantage = (
        rollout['advantage'] - advantage_mean) / advantage_std

    totals = {name: 0.0 for name in (
        'policy_loss', 'value_loss', 'entropy', 'approx_kl',
        'clip_fraction')}
    updates = 0
    stop_early = False
    for _ in range(args.update_epochs):
        permutation = torch.randperm(
            args.batch_size, device=rollout['depth'].device)
        for start in range(0, args.batch_size, args.minibatch_envs):
            indices = permutation[start:start + args.minibatch_envs]
            new_log_prob, entropy, new_value = evaluate_sequence(
                model, rollout, indices)
            mask = rollout['valid'][:, indices]
            old_log_prob = rollout['old_log_prob'][:, indices][mask]
            log_ratio = new_log_prob[mask] - old_log_prob
            ratio = log_ratio.exp()
            advantages = normalized_advantage[:, indices][mask]

            unclipped = -advantages * ratio
            clipped = -advantages * ratio.clamp(
                1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = torch.maximum(unclipped, clipped).mean()

            old_value = rollout['old_value'][:, indices][mask]
            returns = rollout['return'][:, indices][mask]
            value = new_value[mask]
            value_unclipped_loss = (value - returns).square()
            value_clipped = old_value + (value - old_value).clamp(
                -args.clip_coef, args.clip_coef)
            value_clipped_loss = (value_clipped - returns).square()
            value_loss = 0.5 * torch.maximum(
                value_unclipped_loss, value_clipped_loss).mean()
            entropy_loss = entropy[mask].mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > args.clip_coef
                ).float().mean()
            values_to_log = (
                policy_loss, value_loss, entropy_loss, approx_kl,
                clip_fraction)
            for name, value_to_log in zip(totals, values_to_log):
                totals[name] += value_to_log.item()
            updates += 1

            if args.target_kl > 0.0 and approx_kl.item() > args.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {
        name: value / max(updates, 1) for name, value in totals.items()}


def checkpoint_payload(args, model, optimizer, update, global_step):
    return {
        'format': 'icra_ppo_v2',
        'time_limit_bootstrap': True,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': update,
        'global_step': global_step,
        'args': vars(args),
    }


def configure_run_directories(args, checkpoint=None):
    """Resolve unique run directories, or reuse them when resuming."""
    if checkpoint is not None:
        args.save_dir = os.path.dirname(os.path.abspath(args.resume))
        if args.log_dir is None:
            args.log_dir = os.path.join(
                'runs', os.path.basename(args.save_dir))
        args.run_name = os.path.basename(args.save_dir)
        return

    save_root = args.save_dir
    log_root = args.log_dir or 'runs'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'seed{args.seed}_{timestamp}'
    suffix = 1
    while (os.path.exists(os.path.join(save_root, run_name))
           or os.path.exists(os.path.join(log_root, run_name))):
        run_name = f'seed{args.seed}_{timestamp}_{suffix:02d}'
        suffix += 1
    args.run_name = run_name
    args.save_dir = os.path.join(save_root, run_name)
    args.log_dir = os.path.join(log_root, run_name)


def save_checkpoint(args, model, optimizer, update, global_step, final=False):
    os.makedirs(args.save_dir, exist_ok=True)
    suffix = (
        'checkpoint_ppo_final.pth' if final
        else f'checkpoint_ppo_update{update:04d}_steps{global_step}.pth')
    path = os.path.join(args.save_dir, suffix)
    torch.save(
        checkpoint_payload(args, model, optimizer, update, global_step), path)
    latest = os.path.join(args.save_dir, 'checkpoint_ppo_latest.pth')
    torch.save(
        checkpoint_payload(args, model, optimizer, update, global_step),
        latest)
    print(f'Saved PPO checkpoint: {path}')
    return path


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but torch.cuda.is_available() is False')
    set_seed(args.seed)
    cpu_rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    ckpt = None
    start_update = 0
    global_step = 0
    if args.resume:
        print(f'Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get('format') != 'icra_ppo_v2':
            raise ValueError(
                'Only corrected icra_ppo_v2 checkpoints can be resumed. '
                'Legacy v1 checkpoints used zero-value time-limit truncation '
                f'and must be retrained: {args.resume}')
        start_update = int(ckpt['update'])
        global_step = int(ckpt['global_step'])

    configure_run_directories(args, checkpoint=ckpt)
    os.makedirs(args.save_dir, exist_ok=ckpt is not None)
    print(f'Run name: {args.run_name}')
    print(f'Checkpoint directory: {args.save_dir}')
    print(f'TensorBoard directory: {args.log_dir}')

    env = build_env(args, device)
    model = PPOActorCritic(
        dim_obs=6,
        hidden_dim=args.hidden_dim,
        input_w=args.env_width // 2,
        input_h=args.env_height // 2,
        max_v=args.max_speed,
        max_omega=args.max_omega,
        initial_speed=args.cruise_v,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'Resumed PPO training from update {start_update}')

    writer = SummaryWriter(log_dir=args.log_dir)
    param_lines = [
        f'| {key} | {value} |'
        for key, value in sorted(vars(args).items())]
    writer.add_text(
        'Config/TrainParams',
        '| Parameter | Value |\n|---|---|\n' + '\n'.join(param_lines),
    )

    transitions_per_update = args.batch_size * args.rollout_steps
    total_updates = math.ceil(args.total_timesteps / transitions_per_update)
    if start_update >= total_updates:
        raise ValueError(
            f'Checkpoint update {start_update} already reaches the '
            f'requested training budget')

    start_time = time.perf_counter()
    start_global_step = global_step
    progress = tqdm(range(start_update, total_updates), ncols=120)
    active_curriculum_phase = None
    active_obstacle_counts = None
    for update_index in progress:
        update = update_index + 1
        if args.anneal_lr:
            fraction = 1.0 - update_index / total_updates
            optimizer.param_groups[0]['lr'] = args.lr * fraction

        curriculum_phase, density_level, obstacle_counts = \
            obstacle_curriculum_state(args, update_index)
        if obstacle_counts != active_obstacle_counts:
            env.set_obstacle_counts(*obstacle_counts)
            active_obstacle_counts = obstacle_counts
        if curriculum_phase != active_curriculum_phase:
            active_curriculum_phase = curriculum_phase
            if (args.obstacle_curriculum
                    and args.obstacle_curriculum_mode == 'mixed'):
                probabilities = obstacle_curriculum_probabilities(
                    args, curriculum_phase)
                tqdm.write(
                    'Obstacle curriculum mixed phase '
                    f'{curriculum_phase + 1}/3 at update {update_index}: '
                    f'low/mid/high={probabilities}, '
                    f'first_density_level={density_level + 1}, '
                    f'counts={obstacle_counts}'
                )
            elif args.obstacle_curriculum:
                tqdm.write(
                    'Obstacle curriculum stage '
                    f'{curriculum_phase + 1}/4 at update {update_index}: '
                    f'cyl={obstacle_counts[0]}, balls={obstacle_counts[1]}, '
                    f'vox={obstacle_counts[2]}'
                )

        if args.randomize_horizon:
            rollout_steps = int(cpu_rng.integers(
                args.min_timesteps, args.max_timesteps + 1))
        else:
            rollout_steps = args.rollout_steps

        rollout, rollout_metrics = collect_rollout(
            args, env, model, cpu_rng, rollout_steps)
        update_metrics = ppo_update(args, model, optimizer, rollout)
        global_step += args.batch_size * rollout_steps
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        steps_per_second = (
            global_step - start_global_step) / elapsed

        for name, value in rollout_metrics.items():
            writer.add_scalar(f'Rollout/{name}', value, global_step)
        for name, value in update_metrics.items():
            writer.add_scalar(f'PPO/{name}', value, global_step)
        writer.add_scalar(
            'PPO/learning_rate', optimizer.param_groups[0]['lr'],
            global_step)
        writer.add_scalar(
            'Performance/nominal_steps_per_second', steps_per_second,
            global_step)
        writer.add_scalar(
            'Policy/log_std_linear', model.log_std[0].item(), global_step)
        writer.add_scalar(
            'Policy/log_std_angular', model.log_std[1].item(), global_step)

        progress.set_description(
            f"R:{rollout_metrics['return']:.1f}|"
            f"Arr:{rollout_metrics['arrival_rate']:.0%}|"
            f"Col:{rollout_metrics['collision_rate']:.0%}|"
            f"Suc:{rollout_metrics['success_rate']:.0%}|"
            f"D:{rollout_metrics['final_distance']:.1f}|"
            f"KL:{update_metrics['approx_kl']:.3f}"
        )
        if update % args.save_every == 0:
            save_checkpoint(args, model, optimizer, update, global_step)

        if update % 50 == 0:
            p_hist = rollout['p_history']
            final_dist = rollout['terminal_distance']
            min_clr = rollout['min_clearance']
            collided_flag = (min_clr <= 0.0).float()
            worst_score = collided_flag * 1e6 + final_dist
            worst_idx = worst_score.argmax().item()
            plot_trajectory(
                env, p_hist, env.p_target, global_step, writer,
                required_clearance=args.safety_margin,
                batch_idx=worst_idx,
                tag_suffix='_worst',
            )

    save_checkpoint(
        args, model, optimizer, total_updates, global_step, final=True)
    writer.close()


if __name__ == '__main__':
    main()
