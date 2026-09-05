"""Batched benchmark evaluation for the icra_code_0826 navigation policies.

The observation preprocessing, recurrent policy, MPC and motor lag in this
file intentionally match ``train_mpc.py`` so the reported numbers compare
checkpoints rather than slightly different control pipelines.  Model, MPC and
environment settings are read from ``ckpt['args']`` (with explicit
command-line arguments taking precedence) to avoid train/eval mismatch.

The primary task succeeds when the robot first enters ``success_radius``
without any physical collision up to and including that frame.  The PPO
baseline is supported through ``--policy_type ppo``.

Usage:
    python3 evaluate_benchmark.py --checkpoint save/.../checkpoint_mpc_5000.pth
    python3 evaluate_benchmark.py --checkpoint save/.../checkpoint_ppo_*.pth \
        --policy_type ppo --benchmark all
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env_cuda import Env
from model_mpc import Model
from mpc import (
    DifferentiableWaypointMPC,
    depth_to_local_obstacle_points,
    estimate_local_obstacle_velocity,
    estimate_emergency_risk,
)
from ppo_model import PPOActorCritic


DEFAULT_BENCHMARKS = ('open', 'random', 'dense', 'cross', 'dynamic')


BENCHMARKS = {
    # No obstacles: mainly tests goal seeking and braking near the goal.
    'open': {
        'seed_offset': 0,
        'env': {
            'num_cyl': 0, 'num_balls': 0, 'num_vox': 0,
            'dynamic_obstacle_scene_prob': 0.0,
        },
    },
    # Static version of the training obstacle density.
    'random': {
        'seed_offset': 1000,
        'env': {'dynamic_obstacle_scene_prob': 0.0},
    },
    # More clutter than the default training distribution.
    'dense': {
        'seed_offset': 2000,
        'env': {
            'num_cyl': 30, 'num_balls': 18, 'num_vox': 15,
            'dynamic_obstacle_scene_prob': 0.0,
        },
    },
    # A fixed anti-diagonal with moderately dense obstacles.  Training
    # randomizes over all four corner pairs; this freezes one pair while
    # changing the obstacle layout seed.
    'cross': {
        'seed_offset': 3000,
        'env': {
            'num_cyl': 24, 'num_balls': 14, 'num_vox': 12,
            'randomize_start_goal': False,
            'start_pos': (-4.5, 4.5),
            'target_pos': (4.5, -4.5),
            'dynamic_obstacle_scene_prob': 0.0,
        },
    },
    # Training obstacle density, but every scene contains moving obstacles.
    'dynamic': {
        'seed_offset': 4000,
        'env': {'dynamic_obstacle_scene_prob': 1.0},
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--policy_type', choices=('waypoint', 'direct_action', 'ppo'),
        default='waypoint',
        help='Checkpoint architecture; ppo always runs without MPC',
    )
    parser.add_argument(
        '--benchmark', choices=('all', *BENCHMARKS), default='all')
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None,
                        help='Optional JSON file for the aggregated metrics')
    parser.add_argument('--episode_output', default=None,
                        help='Optional CSV containing one row per episode')

    # Evaluation-only overrides; everything else comes from ckpt['args'].
    parser.add_argument('--success_radius', type=float, default=None,
                        help='Defaults to the checkpoint success_radius')
    parser.add_argument('--goal_stop_distance', type=float, default=0.5,
                        help='Command latch radius once the goal is reached')
    parser.add_argument('--safety_margin', type=float, default=0.3,
                        help='Clearance-violation threshold (m)')
    parser.add_argument('--depth_noise_std', type=float, default=0.02)

    args = parser.parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint not found: {checkpoint_path}')
    if args.episodes <= 0 or args.batch_size <= 0 or args.timesteps <= 0:
        parser.error(
            '--episodes, --batch_size and --timesteps must be positive')
    if args.goal_stop_distance < 0.0 or args.safety_margin <= 0.0:
        parser.error(
            'Goal-stop distance must be non-negative and safety margin '
            'positive')
    if args.depth_noise_std < 0.0:
        parser.error('--depth_noise_std must be non-negative')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Defaults mirror configs/xnavdp_dense_corridor_v2.args so partially-saved
# checkpoints still evaluate with sensible settings.
_SAVED_DEFAULTS = {
    'env_width': 64,
    'env_height': 48,
    'grad_decay': 0.8,
    'fov_x_half_tan': 0.82,
    'map_size': 14.0,
    'num_cyl': 27,
    'num_balls': 15,
    'num_vox': 10,
    'robot_radius': 0.24,
    'cyl_radius_min': 0.15,
    'cyl_radius_max': 0.45,
    'cyl_height_min': 0.5,
    'cyl_height_max': 1.5,
    'ball_radius_min': 0.15,
    'ball_radius_max': 0.45,
    'ball_radius_floor': 0.25,
    'initial_yaw_noise': 0.26,
    'randomize_start_goal': True,
    'protected_zone_radius': 1.0,
    'obstacle_layout': 'stratified',
    'obstacle_grid_jitter': 0.35,
    'obstacle_candidate_multiplier': 2.0,
    'dynamic_obstacle_scene_prob': 0.0,
    'dynamic_obstacle_ratio': 0.0,
    'dynamic_obstacle_speed_min': 0.0,
    'dynamic_obstacle_speed_max': 0.0,
    'control_hz': 15.0,
    'dt_noise_std': 0.005,
    'motor_rate_min': 3.0,
    'motor_rate_max': 8.0,
    'max_action_delay_steps': 1,
    'exec_v_scale_std': 0.0,
    'exec_w_scale_std': 0.0,
    'wheel_bias_std': 0.0,
    'max_speed': 2.0,
    'max_omega': 3.0,
    'goal_state_max_distance': 10.0,
    'num_waypoints': 3,
    'hidden_dim': 192,
    'max_forward_step': 1.5,
    'max_lateral_step': 1.0,
    'initial_desired_speed': 1.5,
    'min_desired_speed': 0.1,
    'cruise_v': 1.6,
    'mpc_horizon': 12,
    'mpc_control_lookahead': 3,
    'mpc_max_acc_v': 8.0,
    'mpc_max_acc_omega': 10.0,
    'mpc_max_lateral_acc': 7.0,
    'mpc_track_weight': 8.0,
    'mpc_smooth_weight': 30.0,
    'mpc_initial_velocity_weight': 4.0,
    'mpc_perception_safety': False,
    'mpc_obstacle_clearance': 0.38,
    'mpc_obstacle_temperature': 0.15,
    'mpc_obstacle_refine_steps': 2,
    'obstacle_num_points': 16,
    'obstacle_height_fraction': 0.4,
    'obstacle_depth_quantile': 0.1,
    'obstacle_min_range': 0.2,
    'obstacle_max_range': 6.0,
    'emergency_distance': 1.2,
    'emergency_ttc': 0.7,
    'success_radius': 0.5,
}


def resolve_config(args, checkpoint):
    """Merge saved training args with CLI overrides into one namespace."""
    saved = checkpoint.get('args') or {}
    cfg = argparse.Namespace()
    for key, default in _SAVED_DEFAULTS.items():
        value = saved.get(key)
        setattr(cfg, key, default if value is None else value)
    if args.success_radius is not None:
        cfg.success_radius = args.success_radius
    cfg.goal_stop_distance = args.goal_stop_distance
    cfg.safety_margin = args.safety_margin
    cfg.depth_noise_std = args.depth_noise_std
    cfg.timesteps = args.timesteps
    cfg.device = args.device
    cfg.policy_type = args.policy_type
    cfg.start_pos = (-4.5, -4.5)
    cfg.target_pos = (4.5, 4.5)
    return cfg


def load_checkpoint(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Unsupported checkpoint format: {args.checkpoint}')
    is_ppo_checkpoint = checkpoint.get('format') in {
        'icra_ppo_v1', 'icra_ppo_v2'}
    if args.policy_type == 'ppo' and not is_ppo_checkpoint:
        raise ValueError(
            f'{args.checkpoint} is not a PPO checkpoint '
            f"(format={checkpoint.get('format')!r}); use --policy_type "
            f'waypoint or direct_action')
    if args.policy_type != 'ppo' and is_ppo_checkpoint:
        raise ValueError(
            f'{args.checkpoint} is a PPO checkpoint; pass --policy_type ppo')
    return checkpoint


def load_model(cfg, checkpoint, device):
    if cfg.policy_type == 'ppo':
        model = PPOActorCritic(
            dim_obs=6,
            hidden_dim=cfg.hidden_dim,
            input_w=cfg.env_width // 2,
            input_h=cfg.env_height // 2,
            max_v=cfg.max_speed,
            max_omega=cfg.max_omega,
            initial_speed=min(max(cfg.cruise_v, 1e-3),
                              cfg.max_speed - 1e-3),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model = Model(
            dim_obs=6,
            num_waypoints=cfg.num_waypoints,
            hidden_dim=cfg.hidden_dim,
            input_w=cfg.env_width // 2,
            input_h=cfg.env_height // 2,
            max_forward_step=cfg.max_forward_step,
            max_lateral_step=cfg.max_lateral_step,
            max_speed=cfg.max_speed,
            initial_desired_speed=cfg.initial_desired_speed,
            min_desired_speed=cfg.min_desired_speed,
            direct_action=cfg.policy_type == 'direct_action',
        ).to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model


def build_mpc(cfg, device):
    return DifferentiableWaypointMPC(
        num_waypoints=cfg.num_waypoints,
        horizon=cfg.mpc_horizon,
        control_lookahead=cfg.mpc_control_lookahead,
        max_v=cfg.max_speed,
        max_omega=cfg.max_omega,
        max_acc_v=cfg.mpc_max_acc_v,
        max_acc_omega=cfg.mpc_max_acc_omega,
        max_lateral_acc=cfg.mpc_max_lateral_acc,
        track_weight=cfg.mpc_track_weight,
        smooth_weight=cfg.mpc_smooth_weight,
        initial_velocity_weight=cfg.mpc_initial_velocity_weight,
        perception_safety_enabled=bool(cfg.mpc_perception_safety),
        obstacle_safety_clearance=cfg.mpc_obstacle_clearance,
        obstacle_temperature=cfg.mpc_obstacle_temperature,
        obstacle_refine_steps=cfg.mpc_obstacle_refine_steps,
        collect_diagnostics=False,
    ).to(device)


def build_env(cfg, batch_size, benchmark, device):
    env_kwargs = {
        'map_size': cfg.map_size,
        'num_cyl': cfg.num_cyl,
        'num_balls': cfg.num_balls,
        'num_vox': cfg.num_vox,
        'robot_radius': cfg.robot_radius,
        'cyl_radius_min': cfg.cyl_radius_min,
        'cyl_radius_max': cfg.cyl_radius_max,
        'cyl_height_min': cfg.cyl_height_min,
        'cyl_height_max': cfg.cyl_height_max,
        'ball_radius_min': cfg.ball_radius_min,
        'ball_radius_max': cfg.ball_radius_max,
        'ball_radius_floor': cfg.ball_radius_floor,
        'start_pos': cfg.start_pos,
        'target_pos': cfg.target_pos,
        'randomize_start_goal': cfg.randomize_start_goal,
        'protected_zone_radius': cfg.protected_zone_radius,
        'obstacle_layout': cfg.obstacle_layout,
        'obstacle_grid_jitter': cfg.obstacle_grid_jitter,
        'obstacle_candidate_multiplier': cfg.obstacle_candidate_multiplier,
        'dynamic_obstacle_scene_prob': cfg.dynamic_obstacle_scene_prob,
        'dynamic_obstacle_ratio': cfg.dynamic_obstacle_ratio,
        'dynamic_obstacle_speed_min': cfg.dynamic_obstacle_speed_min,
        'dynamic_obstacle_speed_max': cfg.dynamic_obstacle_speed_max,
    }
    env_kwargs.update(benchmark['env'])
    return Env(
        batch_size, cfg.env_width, cfg.env_height,
        cfg.grad_decay, device,
        fov_x_half_tan=cfg.fov_x_half_tan,
        ground_voxels=True,
        initial_yaw_noise=cfg.initial_yaw_noise,
        **env_kwargs,
    )


def make_state(cfg, env, yaw_obs, v_obs):
    """Goal state in the robot frame, exactly as in train_mpc.py."""
    vec_global = env.p_target[:, :2] - env.p[:, :2]
    cos_th = torch.cos(yaw_obs)
    sin_th = torch.sin(yaw_obs)
    local_x = vec_global[:, 0] * cos_th + vec_global[:, 1] * sin_th
    local_y = vec_global[:, 0] * -sin_th + vec_global[:, 1] * cos_th
    dist_target = torch.sqrt(local_x ** 2 + local_y ** 2)
    max_distance = cfg.goal_state_max_distance
    scale = torch.where(
        dist_target > max_distance,
        max_distance / (dist_target + 1e-6),
        torch.ones_like(dist_target),
    )
    return torch.stack([
        local_x * scale / max_distance,
        local_y * scale / max_distance,
        cos_th,
        sin_th,
        dist_target * scale / max_distance,
        v_obs / cfg.max_speed,
    ], dim=1)


def obstacle_clearance(env):
    """Distance from the robot surface to its closest obstacle surface."""
    return env.signed_clearance(subtract_robot_radius=True)


def _first_true_index(mask):
    """First true state index per episode, or T when never true."""
    state_count = mask.shape[0]
    indices = torch.arange(state_count, device=mask.device)[:, None]
    sentinel = torch.full_like(indices, state_count)
    return torch.where(mask, indices, sentinel).min(dim=0).values


def _masked_time_mean(values, mask):
    """Per-episode temporal mean with a safe zero for empty masks."""
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    if values.ndim == 3:
        numerator = (values * weights).sum(dim=(0, 2))
        denominator = weights.sum(dim=0).squeeze(-1) * values.shape[2]
    else:
        numerator = (values * weights).sum(dim=0)
        denominator = weights.sum(dim=0)
    return numerator / denominator.clamp_min(1.0)


@torch.no_grad()
def rollout_batch(cfg, model, mpc, benchmark, batch_size, seed):
    set_seed(seed)
    device = torch.device(cfg.device)
    env = build_env(cfg, batch_size, benchmark, device)
    env.reset()
    model.reset()

    hidden = None
    motor_actual = torch.zeros((batch_size, 2), device=device)
    response_rate = torch.rand((batch_size, 2), device=device)
    response_rate = response_rate * (
        cfg.motor_rate_max - cfg.motor_rate_min) + cfg.motor_rate_min
    exec_v_scale = (
        1.0 + torch.randn((batch_size, 1), device=device)
        * cfg.exec_v_scale_std
    ).clamp(0.5, 1.5)
    exec_w_scale = (
        1.0 + torch.randn((batch_size, 1), device=device)
        * cfg.exec_w_scale_std
    ).clamp(0.5, 1.5)
    wheel_bias = torch.randn(
        (batch_size, 1), device=device) * cfg.wheel_bias_std
    action_delay_steps = int(torch.randint(
        cfg.max_action_delay_steps + 1, (1,), device=device).item())
    if action_delay_steps:
        action_delay_buffer = [
            torch.zeros_like(motor_actual)
            for _ in range(action_delay_steps)
        ]
        action_delay_cursor = 0
    else:
        action_delay_buffer = None

    positions = [env.p.clone()]
    clearances = [obstacle_clearance(env)]
    actual_speeds = [env.v[:, 0].clone()]
    actual_omegas = [env.omega[:, 2].clone()]
    commands = []
    desired_speeds = []
    elapsed_times = [0.0]
    elapsed = 0.0
    goal_latched = torch.zeros(
        batch_size, dtype=torch.bool, device=device)
    previous_clearance = None
    previous_obstacle_points = None
    previous_obstacle_mask = None

    for _ in range(cfg.timesteps):
        ctl_dt = max(
            0.01, random.normalvariate(
                1.0 / cfg.control_hz, cfg.dt_noise_std))
        # Exact-state observations, identical to train_mpc.py.
        yaw_obs = env.theta[:, 2]
        v_obs = env.v[:, 0]
        w_obs = env.omega[:, 2]

        depth, _ = env.render(ctl_dt)
        if cfg.policy_type == 'waypoint' and cfg.mpc_perception_safety:
            obstacle_points, obstacle_mask = depth_to_local_obstacle_points(
                depth,
                fov_x_half_tan=cfg.fov_x_half_tan,
                num_points=cfg.obstacle_num_points,
                height_fraction=cfg.obstacle_height_fraction,
                depth_quantile=cfg.obstacle_depth_quantile,
                min_range=cfg.obstacle_min_range,
                max_range=cfg.obstacle_max_range,
            )
            obstacle_velocity = estimate_local_obstacle_velocity(
                obstacle_points, obstacle_mask,
                previous_obstacle_points, previous_obstacle_mask,
                v_obs[:, None], w_obs[:, None], ctl_dt,
            )
            emergency_risk, previous_clearance = estimate_emergency_risk(
                depth, previous_clearance, ctl_dt,
                robot_radius=env.drone_radius,
                emergency_distance=cfg.emergency_distance,
                emergency_ttc=cfg.emergency_ttc,
            )
            previous_obstacle_points = obstacle_points
            previous_obstacle_mask = obstacle_mask
        else:
            obstacle_points = obstacle_mask = obstacle_velocity = None
            emergency_risk = None

        depth_inv = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
        if cfg.depth_noise_std > 0.0:
            depth_inv = depth_inv + (
                torch.randn_like(depth_inv) * cfg.depth_noise_std)
        depth_input = F.max_pool2d(depth_inv, 2, 2)
        state = make_state(cfg, env, yaw_obs, v_obs)

        distance_to_goal = torch.linalg.vector_norm(
            env.p_target[:, :2] - env.p[:, :2], dim=1)
        goal_latched = goal_latched | (
            distance_to_goal <= cfg.goal_stop_distance)

        if cfg.policy_type == 'waypoint':
            waypoints, policy_desired_speed, hidden = model(
                depth_input, state, hidden)
            command, _ = mpc(
                waypoints,
                policy_desired_speed,
                current_speed=v_obs[:, None],
                dt=ctl_dt,
                current_omega=w_obs[:, None],
                emergency_risk=emergency_risk,
                obstacle_points=obstacle_points,
                obstacle_mask=obstacle_mask,
                obstacle_velocity=obstacle_velocity,
            )
            desired_speed = policy_desired_speed
        elif cfg.policy_type == 'ppo':
            command, hidden = model.deterministic_action(
                depth_input, state, hidden)
            desired_speed = command[:, 0:1]
        else:
            raw_action, hidden = model(depth_input, state, hidden)
            v_cmd = cfg.max_speed * torch.sigmoid(raw_action[:, 0:1])
            omega_cmd = cfg.max_omega * torch.tanh(raw_action[:, 1:2])
            desired_speed = v_cmd
            command = torch.cat([v_cmd, omega_cmd], dim=1)

        desired_speed = torch.where(
            goal_latched[:, None], torch.zeros_like(desired_speed),
            desired_speed)
        command = torch.where(
            goal_latched[:, None], torch.zeros_like(command), command)

        # Actuator lag, identical to train_mpc.rollout.
        if action_delay_buffer is None:
            delayed_command = command
        else:
            delayed_command = action_delay_buffer[action_delay_cursor]
            action_delay_buffer[action_delay_cursor] = command
            action_delay_cursor = (
                action_delay_cursor + 1) % action_delay_steps
        motor_target = torch.cat([
            (delayed_command[:, 0:1] * exec_v_scale).clamp(
                0.0, cfg.max_speed),
            (
                delayed_command[:, 1:2] * exec_w_scale
                + delayed_command[:, 0:1] * wheel_bias
            ).clamp(-cfg.max_omega, cfg.max_omega),
        ], dim=1)
        motor_alpha = torch.exp(-response_rate * ctl_dt)
        motor_actual = (
            motor_alpha * motor_actual + (1.0 - motor_alpha) * motor_target)
        env.run(motor_actual, ctl_dt)

        positions.append(env.p.clone())
        clearances.append(obstacle_clearance(env))
        actual_speeds.append(env.v[:, 0].clone())
        actual_omegas.append(env.omega[:, 2].clone())
        commands.append(command)
        desired_speeds.append(desired_speed)
        elapsed += ctl_dt
        elapsed_times.append(elapsed)

    return _compute_metrics(
        cfg, batch_size, positions, clearances, actual_speeds,
        actual_omegas, commands, desired_speeds, elapsed_times, env)


def _compute_metrics(
    cfg, batch_size, positions, clearances, actual_speeds, actual_omegas,
    commands, desired_speeds, elapsed_times, env,
):
    device = env.p.device
    position_stack = torch.stack(positions)
    clearance_stack = torch.stack(clearances)
    command_stack = torch.stack(commands)
    actual_speed_stack = torch.stack(actual_speeds)
    actual_omega_stack = torch.stack(actual_omegas)
    desired_speed_stack = torch.stack(desired_speeds).squeeze(-1)
    distance_stack = torch.linalg.vector_norm(
        env.p_target[None, :, :2] - position_stack[:, :, :2], dim=-1)

    final_distance = distance_stack[-1]
    min_clearance = clearance_stack.min(dim=0).values
    stop_history = distance_stack <= cfg.goal_stop_distance
    stop_latched_history = stop_history.to(torch.int64).cumsum(dim=0) > 0
    already_stopped_before_frame = torch.cat([
        torch.zeros_like(stop_latched_history[:1]),
        stop_latched_history[:-1],
    ], dim=0)
    navigation_active = ~already_stopped_before_frame
    post_arrival_active = already_stopped_before_frame
    navigation_min_clearance = clearance_stack.masked_fill(
        ~navigation_active, torch.inf).min(dim=0).values
    post_arrival_min_clearance = clearance_stack.masked_fill(
        ~post_arrival_active, torch.inf).min(dim=0).values
    post_arrival_exposed = post_arrival_active.any(dim=0)
    reached_history = distance_stack <= cfg.success_radius
    final_reached = reached_history[-1]
    ever_reached = reached_history.any(dim=0)
    time_values = torch.tensor(
        elapsed_times, device=device, dtype=distance_stack.dtype)
    state_count = distance_stack.shape[0]
    transition_count = command_stack.shape[0]
    first_reach_index = _first_true_index(reached_history)
    collision_history = clearance_stack <= 0.0
    first_collision_index = _first_true_index(collision_history)
    reached_event = first_reach_index < state_count
    collision_event = first_collision_index < state_count

    # Primary event semantics: the first arrival succeeds only if no physical
    # collision occurred before or on that frame.  A simultaneous arrival and
    # contact is a collision.  Every task metric below is truncated at this
    # first terminal event, so post-arrival stopping cannot favor one policy.
    task_success = reached_event & (
        first_reach_index < first_collision_index)
    task_collision = collision_event & (
        first_collision_index <= first_reach_index)
    task_timeout = ~(task_success | task_collision)
    terminal_state_index = torch.where(
        task_success,
        first_reach_index,
        torch.where(
            task_collision,
            first_collision_index,
            torch.full_like(first_reach_index, state_count - 1),
        ),
    )
    state_indices = torch.arange(state_count, device=device)[:, None]
    transition_indices = torch.arange(
        transition_count, device=device)[:, None]
    task_state_active = state_indices <= terminal_state_index[None, :]
    task_transition_active = (
        transition_indices < terminal_state_index[None, :])
    task_min_clearance = clearance_stack.masked_fill(
        ~task_state_active, torch.inf).min(dim=0).values
    task_clearance_violation = task_min_clearance <= cfg.safety_margin
    task_clearance_success = task_success & ~task_clearance_violation
    task_duration = time_values[terminal_state_index]
    task_terminal_distance = distance_stack.gather(
        0, terminal_state_index[None, :]).squeeze(0)

    collision = min_clearance <= 0.0
    clearance_violation = min_clearance <= cfg.safety_margin
    navigation_collision = navigation_min_clearance <= 0.0
    post_arrival_collision = (
        post_arrival_exposed & (post_arrival_min_clearance <= 0.0))
    safe_success = final_reached & ~collision

    segment_length = position_stack[:, :, :2].diff(dim=0).norm(dim=-1)
    path_length = (
        segment_length * task_transition_active.to(segment_length.dtype)
    ).sum(dim=0)
    shortest_path = distance_stack[0]
    spl = task_success.to(path_length.dtype) * shortest_path / torch.maximum(
        shortest_path, path_length.clamp_min(1e-6))

    actual_speed_active = actual_speed_stack[1:]
    actual_omega_active = actual_omega_stack[1:]
    linear_speed = _masked_time_mean(
        actual_speed_active, task_transition_active)
    abs_omega = _masked_time_mean(
        actual_omega_active.abs(), task_transition_active)
    desired_speed_mean = _masked_time_mean(
        desired_speed_stack, task_transition_active)
    if transition_count > 1:
        smoothness_step = command_stack.diff(dim=0).square().mean(dim=2)
        smoothness_active = (
            torch.arange(transition_count - 1, device=device)[:, None]
            < (terminal_state_index - 1).clamp_min(0)[None, :]
        )
        smoothness = _masked_time_mean(smoothness_step, smoothness_active)
    else:
        smoothness = torch.zeros(batch_size, device=device)

    finite_clearance = torch.isfinite(min_clearance)
    finite_task_clearance = torch.isfinite(task_min_clearance)
    success_float = task_success.to(path_length.dtype)
    aggregate = {
        'episodes': batch_size,
        'final_reached': final_reached.sum().item(),
        'ever_reached': ever_reached.sum().item(),
        'goal_departed': (ever_reached & ~final_reached).sum().item(),
        'time_to_goal_sum': task_duration[task_success].sum().item(),
        'time_to_goal_count': task_success.sum().item(),
        'task_success': task_success.sum().item(),
        'task_collision': task_collision.sum().item(),
        'task_timeout': task_timeout.sum().item(),
        'task_clearance_success': task_clearance_success.sum().item(),
        'task_clearance_violation': task_clearance_violation.sum().item(),
        'safe_success': safe_success.sum().item(),
        'collision': collision.sum().item(),
        'clearance_violation': clearance_violation.sum().item(),
        'navigation_collision': navigation_collision.sum().item(),
        'post_arrival_collision': post_arrival_collision.sum().item(),
        'post_arrival_exposed': post_arrival_exposed.sum().item(),
        'final_distance_sum': final_distance.sum().item(),
        'task_terminal_distance_sum': task_terminal_distance.sum().item(),
        'task_duration_sum': task_duration.sum().item(),
        'path_length_sum': path_length.sum().item(),
        'successful_path_length_sum': (
            path_length * success_float).sum().item(),
        'successful_path_length_count': task_success.sum().item(),
        'spl_sum': spl.sum().item(),
        'smoothness_sum': smoothness.sum().item(),
        'linear_speed_sum': linear_speed.sum().item(),
        'successful_linear_speed_sum': (
            linear_speed * success_float).sum().item(),
        'successful_linear_speed_count': task_success.sum().item(),
        'abs_omega_sum': abs_omega.sum().item(),
        'desired_speed_sum': desired_speed_mean.sum().item(),
        'min_clearance_sum': min_clearance[finite_clearance].sum().item(),
        'min_clearance_count': finite_clearance.sum().item(),
        'min_clearance_min': (
            min_clearance[finite_clearance].min().item()
            if finite_clearance.any() else None),
        'task_min_clearance_sum': (
            task_min_clearance[finite_task_clearance].sum().item()),
        'task_min_clearance_count': finite_task_clearance.sum().item(),
        'task_min_clearance_min': (
            task_min_clearance[finite_task_clearance].min().item()
            if finite_task_clearance.any() else None),
    }
    episode_metrics = {
        'success': task_success,
        'collision': task_collision,
        'timeout': task_timeout,
        'arrival_time': torch.where(
            task_success, task_duration,
            torch.full_like(task_duration, torch.nan)),
        'episode_duration': task_duration,
        'path_length': path_length,
        'shortest_path': shortest_path,
        'spl': spl,
        'smoothness': smoothness,
        'linear_speed': linear_speed,
        'abs_omega': abs_omega,
        'minimum_clearance': task_min_clearance,
        'terminal_distance': task_terminal_distance,
    }
    episode_metrics = {
        key: value.detach().cpu().tolist()
        for key, value in episode_metrics.items()
    }
    return aggregate, episode_metrics


def merge_metrics(total, batch):
    if not total:
        return dict(batch)
    for key, value in batch.items():
        if key in ('min_clearance_min', 'task_min_clearance_min'):
            if value is not None:
                total[key] = (
                    value if total[key] is None
                    else min(total[key], value))
        else:
            total[key] += value
    return total


def finalize_metrics(total):
    episodes = total['episodes']
    ever_reached = total['ever_reached']
    successes = total['task_success']
    return {
        'episodes': episodes,
        'success_rate': total['task_success'] / episodes,
        'collision_rate': total['task_collision'] / episodes,
        'timeout_rate': total['task_timeout'] / episodes,
        'goal_reach_rate': ever_reached / episodes,
        'final_goal_hold_rate': (
            total['final_reached'] / ever_reached if ever_reached else 0.0),
        'goal_departure_rate': (
            total['goal_departed'] / ever_reached if ever_reached else 0.0),
        'time_to_goal_mean': (
            total['time_to_goal_sum'] / total['time_to_goal_count']
            if total['time_to_goal_count'] else None),
        'task_clearance_success_rate': (
            total['task_clearance_success'] / episodes),
        'task_clearance_violation_rate': (
            total['task_clearance_violation'] / episodes),
        'safe_success_rate': total['safe_success'] / episodes,
        'full_horizon_collision_rate': total['collision'] / episodes,
        'clearance_violation_rate': total['clearance_violation'] / episodes,
        'navigation_collision_rate': (
            total['navigation_collision'] / episodes),
        'post_arrival_collision_rate': (
            total['post_arrival_collision'] / total['post_arrival_exposed']
            if total['post_arrival_exposed'] else None),
        'final_distance_mean': total['final_distance_sum'] / episodes,
        'task_terminal_distance_mean': (
            total['task_terminal_distance_sum'] / episodes),
        'episode_duration_mean': total['task_duration_sum'] / episodes,
        'path_length_mean': total['path_length_sum'] / episodes,
        'successful_path_length_mean': (
            total['successful_path_length_sum'] / successes
            if successes else None),
        'spl': total['spl_sum'] / episodes,
        'smoothness_mean': total['smoothness_sum'] / episodes,
        'linear_speed_mean': total['linear_speed_sum'] / episodes,
        'successful_linear_speed_mean': (
            total['successful_linear_speed_sum'] / successes
            if successes else None),
        'abs_omega_mean': total['abs_omega_sum'] / episodes,
        'desired_speed_mean': total['desired_speed_sum'] / episodes,
        'min_clearance_mean': (
            total['min_clearance_sum'] / total['min_clearance_count']
            if total['min_clearance_count'] else None),
        'min_clearance_min': total['min_clearance_min'],
        'task_min_clearance_mean': (
            total['task_min_clearance_sum']
            / total['task_min_clearance_count']
            if total['task_min_clearance_count'] else None),
        'task_min_clearance_min': total['task_min_clearance_min'],
    }


def format_optional(value):
    return 'n/a' if value is None else f'{value:.2f}'


def evaluate_checkpoint(args, model=None, cfg=None):
    """Run all requested benchmarks for one checkpoint.

    Returns ``(results, episode_records)`` where ``results`` maps benchmark
    name -> finalized metrics.  ``model``/``cfg`` may be supplied by callers
    (e.g. the comparison script) that already loaded the checkpoint.
    """
    device = torch.device(args.device)
    if model is None or cfg is None:
        checkpoint = load_checkpoint(args, device)
        cfg = cfg or resolve_config(args, checkpoint)
        model = model or load_model(cfg, checkpoint, device)
    mpc = None
    if cfg.policy_type == 'waypoint':
        mpc = build_mpc(cfg, device)
        mpc.eval()

    benchmark_names = (
        list(DEFAULT_BENCHMARKS) if args.benchmark == 'all'
        else [args.benchmark])
    results = {}
    episode_records = []
    for name in benchmark_names:
        total = {}
        completed = 0
        batch_index = 0
        while completed < args.episodes:
            batch_size = min(args.batch_size, args.episodes - completed)
            seed = args.seed + BENCHMARKS[name]['seed_offset'] + batch_index
            batch, per_episode = rollout_batch(
                cfg, model, mpc, BENCHMARKS[name], batch_size, seed)
            total = merge_metrics(total, batch)
            for local_index in range(batch_size):
                record = {
                    key: values[local_index]
                    for key, values in per_episode.items()
                }
                record.update({
                    'benchmark': name,
                    'seed': seed,
                    'episode_id': completed + local_index,
                })
                episode_records.append(record)
            completed += batch_size
            batch_index += 1
        results[name] = finalize_metrics(total)
    return results, episode_records


def print_summary_line(name, metrics):
    print(
        f'{name:>8} | N {metrics["episodes"]:4d} | '
        f'success {metrics["success_rate"]:.3f} | '
        f'col {metrics["collision_rate"]:.3f} | '
        f'timeout {metrics["timeout_rate"]:.3f} | '
        f'spl {metrics["spl"]:.3f} | '
        f'tgoal {format_optional(metrics["time_to_goal_mean"])} | '
        f'final_dist {metrics["final_distance_mean"]:.2f} | '
        f'path {format_optional(metrics["successful_path_length_mean"])} | '
        f'min_clear {format_optional(metrics["task_min_clearance_min"])} | '
        f'v/w {metrics["linear_speed_mean"]:.2f}/'
        f'{metrics["abs_omega_mean"]:.2f}'
    )


def write_episode_csv(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'benchmark', 'seed', 'episode_id', 'success', 'collision',
        'timeout', 'arrival_time', 'episode_duration', 'path_length',
        'shortest_path', 'spl', 'smoothness', 'linear_speed',
        'abs_omega', 'minimum_clearance', 'terminal_distance',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote per-episode metrics to {output_path}')


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but torch.cuda.is_available() is False')
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args, device)
    cfg = resolve_config(args, checkpoint)
    model = load_model(cfg, checkpoint, device)

    saved = checkpoint.get('args') or {}
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Policy: {args.policy_type} | '
          f'max_speed {cfg.max_speed} | max_omega {cfg.max_omega} | '
          f'mpc_perception_safety {bool(cfg.mpc_perception_safety)} | '
          f'success_radius {cfg.success_radius}')
    if cfg.policy_type == 'waypoint':
        print(f'Waypoints {cfg.num_waypoints} | horizon {cfg.mpc_horizon} | '
              f'lookahead {cfg.mpc_control_lookahead}')
    elif cfg.policy_type == 'ppo':
        print(f'PPO update {checkpoint.get("update")} | '
              f'global_step {checkpoint.get("global_step")}')
    print(f'Env: map {cfg.map_size} | obstacles '
          f'{cfg.num_cyl}/{cfg.num_balls}/{cfg.num_vox} | '
          f'robot_r {cfg.robot_radius} | '
          f'randomize_start_goal {bool(cfg.randomize_start_goal)} | '
          f'layout {cfg.obstacle_layout}')
    if saved:
        print(f'Saved training config keys: {len(saved)}')

    results, episode_records = evaluate_checkpoint(args, model=model, cfg=cfg)
    for name, metrics in results.items():
        print_summary_line(name, metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2), encoding='utf-8')
        print(f'Wrote metrics to {output_path}')
    if args.episode_output:
        write_episode_csv(args.episode_output, episode_records)


if __name__ == '__main__':
    main()
