"""Reproducible inference benchmark for the waypoint-policy + MPC model.

The benchmark deliberately reuses the deployment pipeline (depth -> policy ->
MPC -> actuator lag -> dynamics), but does not reuse training losses as metrics.
It evaluates five reproducible scene distributions: open, random, dense,
randomized diagonal cross, and dynamic obstacles.

Example:
    python3 benchmark.py --checkpoint save/latest.pth
    python3 benchmark.py --checkpoint save/latest.pth \
        --seeds 0 1 2 3 4 --episodes 1024 --output-dir benchmark_results/latest
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env_cuda import Env
from model_mpc import Model
from mpc import (
    DifferentiableWaypointMPC,
    depth_to_local_obstacle_points,
    estimate_emergency_risk,
    estimate_local_obstacle_velocity,
)


SCENES = {
    "open": {"seed_offset": 0},
    "random": {"seed_offset": 1000},
    "dense": {"seed_offset": 2000},
    "cross": {"seed_offset": 3000},
    "dynamic": {"seed_offset": 4000},
}


@dataclass
class EpisodeResult:
    scene: str
    seed: int
    episode: int
    success: bool
    collision: bool
    reached_goal: bool
    arrival_time_s: float | None
    path_length_m: float
    shortest_path_m: float
    spl: float
    final_distance_m: float
    min_body_clearance_m: float | None
    mean_speed_mps: float
    command_variation: float


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark a Diff-Wheelbot waypoint-policy checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="benchmark_results/latest")
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES),
                        default=list(SCENES))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=1024,
                        help="Episodes per scene and seed.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--dt-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-noise-std", type=float, default=0.02,
                        help="Std. dev. of noise after inverse-depth encoding.")
    parser.add_argument("--arrival-tolerance", type=float, default=1.0)
    parser.add_argument("--motor-rate-min", type=float, default=3.0)
    parser.add_argument("--motor-rate-max", type=float, default=8.0)
    parser.add_argument("--max-action-delay-steps", type=int, default=0)
    parser.add_argument("--exec-v-scale-std", type=float, default=0.0)
    parser.add_argument("--exec-w-scale-std", type=float, default=0.0)
    parser.add_argument("--wheel-bias-std", type=float, default=0.0)

    # Policy architecture. These must match the training checkpoint.
    parser.add_argument("--env-width", type=int, default=64)
    parser.add_argument("--env-height", type=int, default=48)
    parser.add_argument("--fov-x-half-tan", type=float, default=0.82)
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument("--start-coordinate", type=float, default=8.0,
                        help="Absolute x/y corner coordinate for start/goal.")
    parser.add_argument("--protected-zone-radius", type=float, default=2.0,
                        help="Obstacle-free radius around start and goal.")
    parser.add_argument("--random-num-cyl", type=int, default=20)
    parser.add_argument("--random-num-balls", type=int, default=12)
    parser.add_argument("--random-num-vox", type=int, default=10)
    parser.add_argument("--dense-num-cyl", type=int, default=30)
    parser.add_argument("--dense-num-balls", type=int, default=18)
    parser.add_argument("--dense-num-vox", type=int, default=15)
    parser.add_argument("--dynamic-obstacle-ratio", type=float, default=0.3)
    parser.add_argument("--dynamic-obstacle-speed-min", type=float, default=0.2)
    parser.add_argument("--dynamic-obstacle-speed-max", type=float, default=1.0)
    parser.add_argument("--num-waypoints", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-forward-step", type=float, default=1.5)
    parser.add_argument("--max-lateral-step", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--max-omega", type=float, default=3.0)
    parser.add_argument("--initial-desired-speed", type=float, default=3.0)
    parser.add_argument("--min-desired-speed", type=float, default=0.15)

    # MPC and optional perception-safety configuration.
    parser.add_argument("--mpc-horizon", type=int, default=12)
    parser.add_argument("--mpc-control-lookahead", type=int, default=3)
    parser.add_argument("--mpc-max-acc-v", type=float, default=8.0)
    parser.add_argument("--mpc-max-acc-omega", type=float, default=8.0)
    parser.add_argument("--mpc-max-lateral-acc", type=float, default=8.0)
    parser.add_argument("--mpc-track-weight", type=float, default=8.0)
    parser.add_argument("--mpc-smooth-weight", type=float, default=30.0)
    parser.add_argument("--mpc-initial-velocity-weight", type=float, default=4.0)
    parser.add_argument("--mpc-perception-safety", type=str2bool, default=False)
    parser.add_argument("--mpc-obstacle-clearance", type=float, default=0.30)
    parser.add_argument("--mpc-obstacle-temperature", type=float, default=0.15)
    parser.add_argument("--mpc-obstacle-refine-steps", type=int, default=2)
    parser.add_argument("--obstacle-num-points", type=int, default=16)
    parser.add_argument("--obstacle-height-fraction", type=float, default=0.4)
    parser.add_argument("--obstacle-depth-quantile", type=float, default=0.1)
    parser.add_argument("--obstacle-min-range", type=float, default=0.2)
    parser.add_argument("--obstacle-max-range", type=float, default=6.0)
    parser.add_argument("--emergency-distance", type=float, default=1.5)
    parser.add_argument("--emergency-ttc", type=float, default=0.8)
    return parser


def validate_args(args):
    if args.episodes < 1 or args.batch_size < 1 or args.timesteps < 1:
        raise ValueError("episodes, batch-size and timesteps must be positive")
    if args.control_hz <= 0.0:
        raise ValueError("control-hz must be positive")
    if args.dt_noise_std < 0.0:
        raise ValueError("dt-noise-std must be non-negative")
    if args.motor_rate_min <= 0.0 or args.motor_rate_max < args.motor_rate_min:
        raise ValueError("require 0 < motor-rate-min <= motor-rate-max")
    if args.arrival_tolerance <= 0.0 or args.depth_noise_std < 0.0:
        raise ValueError("invalid arrival tolerance or depth noise")
    actuator_noise_stds = (
        args.exec_v_scale_std,
        args.exec_w_scale_std,
        args.wheel_bias_std,
    )
    if any(value < 0.0 for value in actuator_noise_stds):
        raise ValueError("actuator-noise stds must be non-negative")
    if args.max_action_delay_steps < 0:
        raise ValueError("max-action-delay-steps must be non-negative")
    counts = (
        args.random_num_cyl, args.random_num_balls, args.random_num_vox,
        args.dense_num_cyl, args.dense_num_balls, args.dense_num_vox,
    )
    if (args.map_size <= 0.0 or args.start_coordinate <= 0.0
            or args.protected_zone_radius < 0.0 or min(counts) < 0):
        raise ValueError("invalid map geometry or obstacle count")
    if not 0.0 <= args.dynamic_obstacle_ratio <= 1.0:
        raise ValueError("dynamic-obstacle-ratio must be in [0, 1]")
    if (args.dynamic_obstacle_speed_min < 0.0
            or args.dynamic_obstacle_speed_max
            < args.dynamic_obstacle_speed_min):
        raise ValueError("invalid dynamic obstacle speed range")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    args.checkpoint = str(checkpoint)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    ).to(device)


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("checkpoint must contain a PyTorch state_dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint/model configuration mismatch; pass the architecture "
            "flags used during training"
        ) from exc
    return checkpoint.get("iter") if isinstance(checkpoint, dict) else None


def scene_env_kwargs(args, scene):
    common = {
        "map_size": args.map_size,
        "num_cyl": args.random_num_cyl,
        "num_balls": args.random_num_balls,
        "num_vox": args.random_num_vox,
        "start_pos": (-args.start_coordinate, -args.start_coordinate),
        "target_pos": (args.start_coordinate, args.start_coordinate),
        "randomize_start_goal": False,
        "protected_zone_radius": args.protected_zone_radius,
        "dynamic_obstacle_scene_prob": 0.0,
        "dynamic_obstacle_ratio": args.dynamic_obstacle_ratio,
        "dynamic_obstacle_speed_min": args.dynamic_obstacle_speed_min,
        "dynamic_obstacle_speed_max": args.dynamic_obstacle_speed_max,
    }
    if scene == "open":
        common.update(num_cyl=0, num_balls=0, num_vox=0)
    elif scene == "dense":
        common.update(
            num_cyl=args.dense_num_cyl,
            num_balls=args.dense_num_balls,
            num_vox=args.dense_num_vox,
        )
    elif scene == "cross":
        common["randomize_start_goal"] = True
    elif scene == "dynamic":
        common["dynamic_obstacle_scene_prob"] = 1.0
    elif scene != "random":
        raise ValueError(f"unknown scene: {scene}")
    return common


def make_env(args, batch_size, device, scene, dynamic_seed):
    env = Env(batch_size, args.env_width, args.env_height, 1.0, device,
              fov_x_half_tan=args.fov_x_half_tan, ground_voxels=True,
              diff_nearest_pt=False, dynamic_obstacle_seed=dynamic_seed,
              **scene_env_kwargs(args, scene))
    env.reset()
    return env


def goal_state(env, xy_obs, yaw_obs, v_obs):
    vec_global = env.p_target[:, :2] - xy_obs
    cos_th = torch.cos(yaw_obs)
    sin_th = torch.sin(yaw_obs)
    local_x = vec_global[:, 0] * cos_th + vec_global[:, 1] * sin_th
    local_y = -vec_global[:, 0] * sin_th + vec_global[:, 1] * cos_th
    distance = torch.sqrt(local_x.square() + local_y.square())
    scale = torch.where(distance > 10.0, 10.0 / distance.clamp_min(1e-6),
                        torch.ones_like(distance))
    local_x, local_y, distance = local_x * scale, local_y * scale, distance * scale
    state = torch.stack((local_x / 10.0, local_y / 10.0, cos_th, sin_th,
                         distance / 10.0, v_obs), dim=1)
    return state


def policy_step(args, env, model, mpc, hidden, perception_state, dt,
                xy_obs, yaw_obs, v_obs, w_obs):
    depth, _ = env.render(dt)
    encoded_depth = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
    if args.depth_noise_std:
        encoded_depth = encoded_depth + torch.randn_like(encoded_depth) \
            * args.depth_noise_std
    depth_input = F.max_pool2d(encoded_depth, 2, 2)

    prev_points, prev_valid, prev_clearance = perception_state
    if args.mpc_perception_safety:
        points, valid = depth_to_local_obstacle_points(
            depth, args.fov_x_half_tan, args.obstacle_num_points,
            args.obstacle_height_fraction, args.obstacle_depth_quantile,
            args.obstacle_min_range, args.obstacle_max_range)
        velocity = estimate_local_obstacle_velocity(
            points, valid, prev_points, prev_valid, v_obs[:, None],
            w_obs[:, None], dt)
        emergency, clearance = estimate_emergency_risk(
            depth, prev_clearance, dt, robot_radius=env.drone_radius,
            emergency_distance=args.emergency_distance,
            emergency_ttc=args.emergency_ttc)
        perception_state = (points, valid, clearance)
    else:
        points = valid = velocity = emergency = None

    waypoints, desired_speed, hidden = model(
        depth_input, goal_state(env, xy_obs, yaw_obs, v_obs), hidden)
    command, _ = mpc(
        waypoints, desired_speed, v_obs[:, None], dt,
        current_omega=w_obs[:, None], emergency_risk=emergency,
        obstacle_points=points, obstacle_mask=valid,
        obstacle_velocity=velocity)
    return command, hidden, perception_state


def run_batch(args, model, mpc, device, scene, seed, episode_offset,
              batch_size):
    dynamic_seed = seed + SCENES[scene]["seed_offset"] + episode_offset + 104729
    env = make_env(args, batch_size, device, scene, dynamic_seed)
    hidden = None
    motor_actual = torch.zeros((batch_size, 2), device=device)
    response_rate = torch.rand((batch_size, 2), device=device) \
        * (args.motor_rate_max - args.motor_rate_min) + args.motor_rate_min
    exec_v_scale = (
        1.0 + torch.randn((batch_size, 1), device=device)
        * args.exec_v_scale_std
    ).clamp(0.5, 1.5)
    exec_w_scale = (
        1.0 + torch.randn((batch_size, 1), device=device)
        * args.exec_w_scale_std
    ).clamp(0.5, 1.5)
    wheel_bias = (
        torch.randn((batch_size, 1), device=device) * args.wheel_bias_std
    )
    # Host-side draw avoids a GPU->CPU sync just to pick the action delay.
    action_delay_steps = min(
        int(random.random() * (args.max_action_delay_steps + 1)),
        args.max_action_delay_steps,
    )
    action_delay_buffer = [
        torch.zeros_like(motor_actual) for _ in range(action_delay_steps + 1)
    ]
    perception_state = (None, None, None)

    initial_position = env.p[:, :2].clone()
    shortest_path = (env.p_target[:, :2] - initial_position).norm(dim=-1)
    previous_position = initial_position
    path_length = torch.zeros(batch_size, device=device)
    speed_sum = torch.zeros(batch_size, device=device)
    evaluated_steps = torch.zeros(batch_size, device=device)
    min_body_clearance = torch.full((batch_size,), float("inf"), device=device)
    first_arrival = torch.full((batch_size,), args.timesteps + 1,
                               dtype=torch.long, device=device)
    first_collision = torch.full_like(first_arrival, args.timesteps + 1)
    variation_sum = torch.zeros(batch_size, device=device)
    variation_count = torch.zeros(batch_size, device=device)
    previous_command = None
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    terminal_distance = torch.full((batch_size,), float("nan"), device=device)
    arrival_time = torch.full((batch_size,), float("nan"), device=device)
    elapsed_time = 0.0

    for step in range(args.timesteps):
        dt = max(0.01, random.normalvariate(
            1.0 / args.control_hz, args.dt_noise_std))
        elapsed_time += dt
        # Exact-state observations (observation noise removed).
        xy_obs = env.p[:, :2]
        yaw_obs = env.theta[:, 2]
        v_obs = env.v[:, 0]
        w_obs = env.omega[:, 2]
        command, hidden, perception_state = policy_step(
            args, env, model, mpc, hidden, perception_state, dt,
            xy_obs, yaw_obs, v_obs, w_obs)
        action_delay_buffer.append(command)
        delayed_command = action_delay_buffer[-(action_delay_steps + 1)]
        motor_target = torch.cat([
            (delayed_command[:, 0:1] * exec_v_scale).clamp(
                0.0, args.max_speed),
            (
                delayed_command[:, 1:2] * exec_w_scale
                + delayed_command[:, 0:1] * wheel_bias
            ).clamp(-args.max_omega, args.max_omega),
        ], dim=1)
        alpha = torch.exp(-response_rate * dt)
        motor_actual = (
            alpha * motor_actual + (1.0 - alpha) * motor_target
        )
        env.run(motor_actual, dt)

        position = env.p[:, :2]
        path_length += active * (position - previous_position).norm(dim=-1)
        previous_position = position.clone()
        speed_sum += active * env.v[:, 0]
        evaluated_steps += active
        if previous_command is not None:
            variation_sum += active * (command - previous_command).square().mean(dim=-1)
            variation_count += active
        previous_command = command

        if scene == "open":
            body_clearance = torch.full((batch_size,), float("inf"),
                                        device=device)
        else:
            body_clearance = env.signed_clearance(
                subtract_robot_radius=True)
        min_body_clearance = torch.where(
            active, torch.minimum(min_body_clearance, body_clearance),
            min_body_clearance)
        distance = (env.p_target[:, :2] - position).norm(dim=-1)
        newly_collided = active & (body_clearance <= 0.0)
        # Collision takes precedence if goal entry and contact happen together.
        newly_arrived = active & ~newly_collided \
            & (distance <= args.arrival_tolerance)
        first_arrival[newly_arrived] = step
        arrival_time[newly_arrived] = elapsed_time
        first_collision[newly_collided] = step
        newly_done = newly_arrived | newly_collided
        terminal_distance[newly_done] = distance[newly_done]
        active = active & ~newly_done
        if not active.any():
            break

    horizon_distance = (env.p_target[:, :2] - env.p[:, :2]).norm(dim=-1)
    final_distance = torch.where(active, horizon_distance, terminal_distance)
    reached = first_arrival <= args.timesteps
    collided = first_collision <= args.timesteps
    # Collision on the same step as goal entry invalidates success.
    success = reached & (first_arrival < first_collision)
    spl = success.float() * shortest_path / torch.maximum(shortest_path,
                                                           path_length)
    arrival_time = torch.where(success, arrival_time,
                               torch.full_like(arrival_time, float("nan")))
    command_variation = variation_sum / variation_count.clamp_min(1.0)

    cpu = {name: value.detach().cpu() for name, value in {
        "success": success, "collision": collided, "reached": reached,
        "arrival_time": arrival_time, "path_length": path_length,
        "shortest_path": shortest_path, "spl": spl,
        "final_distance": final_distance,
        "min_body_clearance": min_body_clearance,
        "mean_speed": speed_sum / evaluated_steps.clamp_min(1.0),
        "command_variation": command_variation,
    }.items()}
    return [EpisodeResult(
        scene=scene, seed=seed, episode=episode_offset + i,
        success=bool(cpu["success"][i]), collision=bool(cpu["collision"][i]),
        reached_goal=bool(cpu["reached"][i]),
        arrival_time_s=(float(cpu["arrival_time"][i])
                        if cpu["success"][i] else None),
        path_length_m=float(cpu["path_length"][i]),
        shortest_path_m=float(cpu["shortest_path"][i]),
        spl=float(cpu["spl"][i]),
        final_distance_m=float(cpu["final_distance"][i]),
        min_body_clearance_m=(
            float(cpu["min_body_clearance"][i])
            if torch.isfinite(cpu["min_body_clearance"][i]) else None
        ),
        mean_speed_mps=float(cpu["mean_speed"][i]),
        command_variation=float(cpu["command_variation"][i]),
    ) for i in range(batch_size)]


def wilson_interval(successes, count, z=1.96):
    if count == 0:
        return [None, None]
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / count
                           + z * z / (4.0 * count * count)) / denominator
    return [center - margin, center + margin]


def summarize(results):
    count = len(results)
    successes = sum(row.success for row in results)
    collisions = sum(row.collision for row in results)
    successful_arrivals = [row.arrival_time_s for row in results
                           if row.arrival_time_s is not None]

    def mean(field):
        values = [getattr(row, field) for row in results
                  if getattr(row, field) is not None]
        return float(np.mean(values)) if values else None

    return {
        "episodes": count,
        "success_rate": successes / count,
        "success_rate_ci95": wilson_interval(successes, count),
        "collision_rate": collisions / count,
        "collision_rate_ci95": wilson_interval(collisions, count),
        "spl": mean("spl"),
        "command_variation": mean("command_variation"),
        "mean_speed_mps": mean("mean_speed_mps"),
        "mean_final_distance_m": mean("final_distance_m"),
        "mean_min_body_clearance_m": mean("min_body_clearance_m"),
        "mean_arrival_time_s_success_only": (
            float(np.mean(successful_arrivals)) if successful_arrivals else None
        ),
    }


def write_outputs(output_dir, args, checkpoint_iter, results, elapsed_s):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(row) for row in results]
    with (output_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for scene in args.scenes:
        scene_rows = [row for row in results if row.scene == scene]
        grouped[scene] = {
            "overall": summarize(scene_rows),
            "by_seed": {
                str(seed): summarize([row for row in scene_rows
                                      if row.seed == seed])
                for seed in args.seeds
            },
        }
    payload = {
        "protocol": {
            "checkpoint": args.checkpoint,
            "checkpoint_iter": checkpoint_iter,
            "scenes": args.scenes,
            "scene_seed_offsets": {
                scene: SCENES[scene]["seed_offset"] for scene in args.scenes
            },
            "scene_environment": {
                scene: scene_env_kwargs(args, scene) for scene in args.scenes
            },
            "seeds": args.seeds,
            "episodes_per_scene_seed": args.episodes,
            "timesteps": args.timesteps,
            "control_hz": args.control_hz,
            "dt_noise_std": args.dt_noise_std,
            "depth_noise_std": args.depth_noise_std,
            "max_action_delay_steps": args.max_action_delay_steps,
            "exec_v_scale_std": args.exec_v_scale_std,
            "exec_w_scale_std": args.exec_w_scale_std,
            "wheel_bias_std": args.wheel_bias_std,
            "arrival_tolerance_m": args.arrival_tolerance,
            "collision_definition": "outer-surface signed clearance <= 0",
            "success_definition": "first goal entry before any collision",
            "command_variation_definition": "mean_t(mean_dims((u_t-u_t-1)^2))",
            "device": str(args.device),
            "elapsed_s": elapsed_s,
        },
        "results": grouped,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return payload


def main():
    args = build_parser().parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    set_seed(args.seeds[0])
    model = build_model(args, device)
    mpc = build_mpc(args, device)
    checkpoint_iter = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    mpc.eval()

    results = []
    started = time.perf_counter()
    with torch.no_grad():
        for scene in args.scenes:
            for seed in args.seeds:
                set_seed(seed + SCENES[scene]["seed_offset"])
                completed = 0
                while completed < args.episodes:
                    current_batch = min(args.batch_size, args.episodes - completed)
                    results.extend(run_batch(
                        args, model, mpc, device, scene, seed, completed,
                        current_batch))
                    completed += current_batch
                    print(f"[{scene}] seed={seed} {completed}/{args.episodes}",
                          flush=True)
    elapsed_s = time.perf_counter() - started
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = write_outputs(output_dir, args, checkpoint_iter, results, elapsed_s)

    print(json.dumps(payload["results"], indent=2))
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
