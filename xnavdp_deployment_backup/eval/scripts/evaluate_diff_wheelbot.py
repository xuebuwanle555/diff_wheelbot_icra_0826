"""Evaluate the diff-wheelbot waypoint policy in the X-NavDP IsaacLab scenes.

This runner reuses X-NavDP's scene, Dingo, start/goal samples, termination
logic, video layout, success metric, and SPL calculation.  Only the policy and
controller are replaced by the external depth/goal waypoint model and its MPC.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="eval/config/eval_pointgoal/wheeled_clutter_easy.yaml",
    )
    parser.add_argument(
        "--model_repo",
        default="/home/jesse/ICRA2027/diff_wheelbot_moving_side_repro",
    )
    parser.add_argument("--model_module", default="model")
    parser.add_argument(
        "--use_checkpoint_config",
        action="store_true",
        help="Rebuild the model and MPC from metadata stored in the checkpoint.",
    )
    parser.add_argument("--run_label", default="pcdps-mpc")
    parser.add_argument(
        "--campaign_id",
        default=None,
        help=(
            "Optional fixed output directory name shared by a multi-scene "
            "paper evaluation. When omitted, the legacy timestamped name is used."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Deterministic IsaacLab environment seed.",
    )

    parser.add_argument(
        "--sensor_profile",
        choices=("training", "dingo"),
        default="training",
        help="Use the training-matched 0.25 m/78.7 deg camera or the stock Dingo D455.",
    )
    parser.add_argument(
        "--control_dt",
        type=float,
        default=None,
        help="Policy period; defaults to 1/15 s for training and 0.1 s for Dingo.",
    )

    # Keep the checkpoint's speed-head semantics separate from the execution
    # envelope.  The original model was trained with 4.5/3.8 m/s, even when a
    # downstream comparison chooses a lower MPC velocity limit.
    parser.add_argument("--policy_max_v", type=float, default=4.5)
    parser.add_argument("--policy_cruise_v", type=float, default=3.8)
    parser.add_argument(
        "--policy_min_v", type=float, default=None,
        help="Speed-head floor at deployment. Defaults to the checkpoint's "
             "min_desired_speed when --use_checkpoint_config is set, "
             "otherwise 0.0. The speed head is parameterised as "
             "min + (max - min) * sigmoid, so this must match the training "
             "value or the learned mapping is shifted.",
    )
    parser.add_argument("--max_v", type=float, default=4.5)
    parser.add_argument("--max_omega", type=float, default=3.0)
    parser.add_argument("--goal_stop_distance", type=float, default=1.0)
    parser.add_argument(
        "--goal_state_max_distance",
        type=float,
        default=10.0,
        help=(
            "Clip the point-goal distance to this value and normalize the "
            "goal-state coordinates by the same value. It must match training."
        ),
    )

    # Final paper-main-simple MPC settings.  The angular limits include the
    # controller override used by the current reproducibility configuration.
    parser.add_argument("--mpc_horizon", type=int, default=12)
    parser.add_argument("--mpc_lookahead", type=int, default=3)
    parser.add_argument("--mpc_max_acc_v", type=float, default=10.0)
    parser.add_argument("--mpc_max_acc_omega", type=float, default=14.0)
    parser.add_argument("--mpc_emergency_max_acc_omega", type=float, default=24.0)
    parser.add_argument("--mpc_max_lateral_acc", type=float, default=7.0)
    parser.add_argument("--mpc_track_weight", type=float, default=8.0)
    parser.add_argument("--mpc_smooth_weight", type=float, default=15.0)
    parser.add_argument("--mpc_emergency_smooth_weight", type=float, default=8.0)
    parser.add_argument("--mpc_initial_velocity_weight", type=float, default=4.0)
    parser.add_argument("--mpc_obstacle_points", type=int, default=16)
    parser.add_argument("--mpc_obstacle_height_fraction", type=float, default=0.40)
    parser.add_argument("--mpc_obstacle_depth_quantile", type=float, default=0.10)
    parser.add_argument("--mpc_obstacle_max_range", type=float, default=6.0)
    parser.add_argument("--mpc_obstacle_safety_clearance", type=float, default=0.30)
    parser.add_argument("--mpc_obstacle_temperature", type=float, default=0.15)
    parser.add_argument("--mpc_obstacle_weight", type=float, default=1.0)
    parser.add_argument("--mpc_obstacle_refine_steps", type=int, default=2)
    parser.add_argument("--mpc_obstacle_step_size", type=float, default=0.06)
    parser.add_argument("--mpc_intervention_weight", type=float, default=3.0)
    parser.add_argument("--mpc_obstacle_slowdown_gain", type=float, default=0.35)
    parser.add_argument("--mpc_obstacle_min_speed_scale", type=float, default=0.25)
    parser.add_argument("--mpc_ttc_clearance_inflation", type=float, default=0.15)
    parser.add_argument("--emergency_distance", type=float, default=1.5)
    parser.add_argument("--emergency_ttc", type=float, default=0.8)
    parser.add_argument("--emergency_depth_quantile", type=float, default=0.02)
    parser.add_argument("--emergency_depth_height_fraction", type=float, default=0.40)
    parser.add_argument("--robot_radius", type=float, default=0.15)
    parser.add_argument(
        "--fov_x_half_tan",
        type=float,
        default=None,
        help="Defaults to 0.82 for training profile and 0.980396 for Dingo D455.",
    )
    parser.add_argument(
        "--contact_force_threshold",
        type=float,
        default=6.5,
        help=(
            "Horizontal contact-force threshold in newtons. A collision is "
            "confirmed after --collision_force_consecutive_steps consecutive "
            "samples above this threshold."
        ),
    )
    parser.add_argument(
        "--collision_force_consecutive_steps",
        type=int,
        default=2,
        help="Consecutive horizontal-force threshold crossings required for collision.",
    )
    parser.add_argument(
        "--collision_peak_force_threshold",
        type=float,
        default=13.0,
        help=(
            "Single-sample horizontal-force threshold in newtons for a strong "
            "impact that bypasses consecutive-sample filtering."
        ),
    )
    parser.add_argument(
        "--collision_settle_clear_steps",
        type=int,
        default=3,
        help=(
            "Require this many consecutive contact-free zero-command steps "
            "before policy control and metric timing begin."
        ),
    )
    parser.add_argument(
        "--collision_settle_max_steps",
        type=int,
        default=15,
        help=(
            "Maximum zero-command spawn-settling steps. Persistent contact "
            "rejects the episode instead of being hidden from metrics."
        ),
    )
    prealign_enable_group = parser.add_mutually_exclusive_group()
    prealign_enable_group.add_argument(
        "--prealign_goal",
        action="store_true",
        help="Rotate in place toward the point goal before enabling the learned policy.",
    )
    prealign_enable_group.add_argument(
        "--no_prealign_goal",
        dest="prealign_goal",
        action="store_false",
        help="Disable launch-heading pre-alignment for policy-only ablations.",
    )
    parser.set_defaults(prealign_goal=False)
    parser.add_argument("--prealign_tolerance_deg", type=float, default=10.0)
    parser.add_argument("--prealign_kp", type=float, default=2.0)
    parser.add_argument("--prealign_max_omega", type=float, default=1.0)
    parser.add_argument("--prealign_stable_steps", type=int, default=3)
    parser.add_argument("--prealign_omega_tolerance", type=float, default=0.20)
    parser.add_argument("--prealign_timeout", type=float, default=8.0)
    parser.add_argument(
        "--prealign_safe_clearance",
        type=float,
        default=0.0,
        help=(
            "When positive, do not release pre-alignment toward a blocked "
            "heading. Rotate toward the more open side until the forward "
            "body-width corridor reaches this clearance. Zero preserves the "
            "legacy goal-only pre-alignment."
        ),
    )
    parser.add_argument(
        "--prealign_corridor_half_width",
        type=float,
        default=0.35,
        help="Half width in metres of the forward corridor checked before launch.",
    )
    parser.add_argument(
        "--prealign_clearance_hysteresis",
        type=float,
        default=0.10,
        help="Extra clearance required before obstacle-search pre-alignment releases.",
    )
    parser.add_argument(
        "--prealign_search_omega",
        type=float,
        default=0.70,
        help="Absolute in-place turn rate used to search for a free launch heading.",
    )
    parser.add_argument(
        "--prealign_obstacle_points",
        type=int,
        default=32,
        help="Number of horizontal depth sectors used by safe pre-alignment.",
    )
    parser.add_argument(
        "--prealign_depth_quantile",
        type=float,
        default=0.10,
        help="Robust vertical depth quantile used by safe pre-alignment.",
    )
    parser.add_argument(
        "--prealign_depth_height_fraction",
        type=float,
        default=0.40,
        help="Upper image-height fraction used by safe pre-alignment.",
    )
    terminal_enable_group = parser.add_mutually_exclusive_group()
    terminal_enable_group.add_argument(
        "--terminal_approach",
        action="store_true",
        help=(
            "Latch an analytic goal controller inside --terminal_enter_distance. "
            "The learned policy and MPC are no longer evaluated after latching."
        ),
    )
    terminal_enable_group.add_argument(
        "--no_terminal_approach",
        dest="terminal_approach",
        action="store_false",
        help="Disable the analytic final-approach controller for policy-only ablations.",
    )
    parser.set_defaults(terminal_approach=False)
    parser.add_argument("--terminal_enter_distance", type=float, default=0.70)
    parser.add_argument("--terminal_max_v", type=float, default=0.20)
    parser.add_argument("--terminal_k_v", type=float, default=1.0)
    parser.add_argument("--terminal_k_omega", type=float, default=2.0)
    parser.add_argument("--terminal_max_omega", type=float, default=0.80)
    parser.add_argument("--terminal_heading_stop_deg", type=float, default=25.0)
    parser.add_argument(
        "--terminal_goal_margin",
        type=float,
        default=0.05,
        help=(
            "Aim this far inside the official success radius so linear "
            "slowdown does not asymptotically stop just outside it."
        ),
    )
    parser.add_argument(
        "--terminal_safe_clearance",
        type=float,
        default=0.25,
        help=(
            "Required residual obstacle clearance after travelling the "
            "remaining distance to the official success radius."
        ),
    )
    parser.add_argument(
        "--terminal_corridor_half_width",
        type=float,
        default=0.35,
        help="Half width in metres of the terminal controller depth corridor.",
    )
    parser.add_argument(
        "--hidden_reset_interval",
        type=int,
        default=0,
        help=(
            "Reset the recurrent policy state after this many actual policy "
            "steps; zero disables periodic reset."
        ),
    )
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--record_video",
        dest="record_video",
        action="store_true",
        help="Write one RGB/bird-eye MP4 for every evaluated episode.",
    )
    video_group.add_argument(
        "--no_record_video",
        dest="record_video",
        action="store_false",
        help="Skip MP4 encoding during large metric runs.",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument(
        "--video_front_view",
        choices=("rgb", "depth"),
        default="rgb",
        help=(
            "Left video panel: low-resolution RGB or the exact metric depth "
            "stream used by the policy rendered as grayscale."
        ),
    )
    parser.add_argument(
        "--video_depth_near",
        type=float,
        default=0.2,
        help="Depth mapped to black in the grayscale video panel (metres).",
    )
    parser.add_argument(
        "--video_depth_far",
        type=float,
        default=10.0,
        help="Depth mapped to white in the grayscale video panel (metres).",
    )
    recovery_enable_group = parser.add_mutually_exclusive_group()
    recovery_enable_group.add_argument(
        "--stuck_recovery",
        action="store_true",
        help=(
            "Enable a closed-loop in-place recovery controller when both the "
            "commanded and measured speeds remain low without goal progress."
        ),
    )
    recovery_enable_group.add_argument(
        "--no_stuck_recovery",
        dest="stuck_recovery",
        action="store_false",
        help="Disable the deployment recovery controller for ablation.",
    )
    parser.set_defaults(stuck_recovery=False)
    parser.add_argument("--recovery_command_v_threshold", type=float, default=0.12)
    parser.add_argument("--recovery_actual_v_threshold", type=float, default=0.05)
    parser.add_argument("--recovery_confirm_time", type=float, default=1.5)
    parser.add_argument("--recovery_progress_threshold", type=float, default=0.08)
    parser.add_argument("--recovery_min_goal_distance", type=float, default=1.0)
    parser.add_argument("--recovery_turn_omega", type=float, default=0.70)
    parser.add_argument("--recovery_min_turn_deg", type=float, default=30.0)
    parser.add_argument("--recovery_max_turn_deg", type=float, default=135.0)
    parser.add_argument("--recovery_front_clearance", type=float, default=0.90)
    parser.add_argument("--recovery_strong_clearance", type=float, default=1.20)
    parser.add_argument("--recovery_policy_v_threshold", type=float, default=0.20)
    parser.add_argument("--recovery_corridor_half_width", type=float, default=0.35)
    parser.add_argument("--recovery_goal_bias", type=float, default=0.15)
    parser.add_argument("--recovery_stable_steps", type=int, default=4)
    parser.add_argument("--recovery_omega_tolerance", type=float, default=0.20)
    parser.add_argument("--recovery_cooldown", type=float, default=1.2)
    parser.add_argument("--recovery_max_sweeps", type=int, default=2)
    parser.add_argument(
        "--open_loop_v",
        type=float,
        default=None,
        help="Diagnostic override for a constant linear command.",
    )
    parser.add_argument(
        "--open_loop_omega",
        type=float,
        default=None,
        help="Diagnostic override for a constant angular command.",
    )
    parser.add_argument(
        "--fixed_speed",
        type=float,
        default=None,
        help=(
            "Diagnostic: ignore the learned speed head and feed this constant "
            "value to MPC instead.  The predicted value is still logged as "
            "predicted_desired_speed so you can inspect whether it collapses."
        ),
    )
    parser.add_argument(
        "--speed_mode",
        choices=("learned", "fixed"),
        default="learned",
        help=(
            "Speed-reference source for MPC. "
            "'learned' uses the model speed head; "
            "'fixed' ignores the learned speed head and uses --fixed_speed."
        ),
    )
    parser.add_argument(
        "--stateless_policy",
        action="store_true",
        help=(
            "Diagnostic only: reset the recurrent hidden state before every "
            "policy forward pass. This disables recurrent-state accumulation "
            "while keeping the same GRU/model weights."
        ),
    )
    parser.add_argument(
        "--policy_speed_input_override",
        type=float,
        default=None,
        help=(
            "Diagnostic only: override only the current-speed scalar passed "
            "into the neural policy state. The MPC and robot dynamics must "
            "continue using the real measured current speed."
        ),
    )
    return parser.parse_args()


def yaw_from_wxyz(quaternion):
    """Return world-frame yaw from IsaacLab wxyz quaternions."""
    import torch

    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def write_metrics(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.control_dt is None:
        args.control_dt = 1.0 / 15.0 if args.sensor_profile == "training" else 0.1
    if args.fov_x_half_tan is None:
        args.fov_x_half_tan = 0.82 if args.sensor_profile == "training" else 0.980396
    if not 0.0 < args.policy_cruise_v < args.policy_max_v:
        raise ValueError("policy_cruise_v must be between zero and policy_max_v")
    if (args.policy_min_v is not None
            and not 0.0 <= args.policy_min_v < args.policy_cruise_v):
        raise ValueError(
            "policy_min_v must be non-negative and below policy_cruise_v")
    if args.speed_mode == "fixed" and args.fixed_speed is None:
        raise ValueError(
            "--fixed_speed must be provided when --speed_mode=fixed"
        )
    if args.policy_speed_input_override is not None:
        if not 0.0 <= args.policy_speed_input_override <= args.policy_max_v:
            raise ValueError(
                "--policy_speed_input_override must be in [0, policy_max_v]"
            )
    if args.hidden_reset_interval < 0:
        raise ValueError("hidden_reset_interval must be non-negative")
    if not 0.0 < args.video_depth_near < args.video_depth_far:
        raise ValueError(
            "video depth range must satisfy 0 < video_depth_near < "
            "video_depth_far"
        )
    non_negative_recovery_values = {
        "recovery_command_v_threshold": args.recovery_command_v_threshold,
        "recovery_actual_v_threshold": args.recovery_actual_v_threshold,
        "recovery_progress_threshold": args.recovery_progress_threshold,
        "recovery_min_goal_distance": args.recovery_min_goal_distance,
        "recovery_front_clearance": args.recovery_front_clearance,
        "recovery_strong_clearance": args.recovery_strong_clearance,
        "recovery_policy_v_threshold": args.recovery_policy_v_threshold,
        "recovery_goal_bias": args.recovery_goal_bias,
        "recovery_omega_tolerance": args.recovery_omega_tolerance,
        "recovery_cooldown": args.recovery_cooldown,
    }
    for name, value in non_negative_recovery_values.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if args.recovery_confirm_time <= 0.0:
        raise ValueError("recovery_confirm_time must be positive")
    if not 0.0 < args.recovery_turn_omega <= args.max_omega:
        raise ValueError("recovery_turn_omega must be in (0, max_omega]")
    if not 0.0 < args.recovery_min_turn_deg < args.recovery_max_turn_deg <= 180.0:
        raise ValueError(
            "recovery turn angles must satisfy 0 < min < max <= 180 degrees"
        )
    if args.recovery_strong_clearance < args.recovery_front_clearance:
        raise ValueError(
            "recovery_strong_clearance must be at least recovery_front_clearance"
        )
    if args.collision_settle_clear_steps < 1:
        raise ValueError("collision_settle_clear_steps must be positive")
    if args.contact_force_threshold <= 0.0:
        raise ValueError("contact_force_threshold must be positive")
    if args.collision_force_consecutive_steps < 1:
        raise ValueError("collision_force_consecutive_steps must be positive")
    if args.collision_peak_force_threshold < args.contact_force_threshold:
        raise ValueError(
            "collision_peak_force_threshold must be at least contact_force_threshold"
        )
    if args.collision_settle_max_steps < args.collision_settle_clear_steps:
        raise ValueError(
            "collision_settle_max_steps must be at least "
            "collision_settle_clear_steps"
        )
    if args.recovery_corridor_half_width <= 0.0:
        raise ValueError("recovery_corridor_half_width must be positive")
    if args.recovery_stable_steps < 1 or args.recovery_max_sweeps < 1:
        raise ValueError("recovery_stable_steps and recovery_max_sweeps must be positive")
    if not 0.0 < args.prealign_tolerance_deg < 180.0:
        raise ValueError("prealign_tolerance_deg must be in (0, 180)")
    if args.prealign_kp <= 0.0 or args.prealign_max_omega <= 0.0:
        raise ValueError("prealign_kp and prealign_max_omega must be positive")
    if args.prealign_stable_steps < 1 or args.prealign_timeout <= 0.0:
        raise ValueError("prealign_stable_steps and prealign_timeout must be positive")
    if args.prealign_safe_clearance < 0.0:
        raise ValueError("prealign_safe_clearance must be non-negative")
    if args.prealign_corridor_half_width <= 0.0:
        raise ValueError("prealign_corridor_half_width must be positive")
    if args.prealign_clearance_hysteresis < 0.0:
        raise ValueError("prealign_clearance_hysteresis must be non-negative")
    if not 0.0 < args.prealign_search_omega <= args.prealign_max_omega:
        raise ValueError(
            "prealign_search_omega must be in (0, prealign_max_omega]"
        )
    if args.prealign_obstacle_points < 4:
        raise ValueError("prealign_obstacle_points must be at least 4")
    if not 0.0 < args.prealign_depth_quantile <= 1.0:
        raise ValueError("prealign_depth_quantile must be in (0, 1]")
    if not 0.0 < args.prealign_depth_height_fraction <= 1.0:
        raise ValueError("prealign_depth_height_fraction must be in (0, 1]")
    if (
        args.terminal_approach
        and args.terminal_enter_distance <= args.goal_stop_distance
    ):
        raise ValueError(
            "terminal_enter_distance must be larger than goal_stop_distance"
        )
    if args.terminal_max_v <= 0.0 or args.terminal_max_v > args.max_v:
        raise ValueError("terminal_max_v must be in (0, max_v]")
    if args.terminal_k_v <= 0.0 or args.terminal_k_omega <= 0.0:
        raise ValueError("terminal controller gains must be positive")
    if not 0.0 < args.terminal_max_omega <= args.max_omega:
        raise ValueError("terminal_max_omega must be in (0, max_omega]")
    if not 0.0 < args.terminal_heading_stop_deg < 90.0:
        raise ValueError("terminal_heading_stop_deg must be in (0, 90)")
    if not 0.0 <= args.terminal_goal_margin < args.goal_stop_distance:
        raise ValueError(
            "terminal_goal_margin must be in [0, goal_stop_distance)"
        )
    if args.terminal_safe_clearance < 0.0:
        raise ValueError("terminal_safe_clearance must be non-negative")
    if args.terminal_corridor_half_width <= 0.0:
        raise ValueError("terminal_corridor_half_width must be positive")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, distributed=True)
    simulation_app = app_launcher.app

    import cv2
    import imageio
    import numpy as np
    import torch
    import torch.nn.functional as F

    from eval.config_utils import load_default_config
    from eval.environment import create_environment

    model_repo = Path(args.model_repo).resolve()
    if not model_repo.is_dir():
        raise FileNotFoundError(f"diff-wheelbot model repo not found: {model_repo}")
    if str(model_repo) not in sys.path:
        sys.path.insert(0, str(model_repo))
    model_module = importlib.import_module(args.model_module)
    mpc_module = importlib.import_module("mpc")
    Model = model_module.Model
    DifferentiableWaypointMPC = mpc_module.DifferentiableWaypointMPC
    depth_to_local_obstacle_points = mpc_module.depth_to_local_obstacle_points
    estimate_emergency_risk = mpc_module.estimate_emergency_risk

    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    checkpoint_args = (
        checkpoint.get("args", {})
        if args.use_checkpoint_config and isinstance(checkpoint, dict)
        else {}
    )
    if checkpoint_args:
        args.policy_max_v = float(
            checkpoint_args.get("max_speed", args.policy_max_v)
        )
        args.policy_cruise_v = float(
            checkpoint_args.get(
                "initial_desired_speed", args.policy_cruise_v
            )
        )
        # The speed head is parameterised as min + (max - min) * sigmoid,
        # so the deployment floor must match the training value or the
        # learned mapping is silently shifted. An explicit --policy_min_v
        # on the command line still wins.
        if args.policy_min_v is None:
            args.policy_min_v = float(
                checkpoint_args.get("min_desired_speed", 0.0)
            )
        args.fov_x_half_tan = float(
            checkpoint_args.get("fov_x_half_tan", args.fov_x_half_tan)
        )
    if args.policy_min_v is None:
        args.policy_min_v = 0.0
    if not 0.0 <= args.policy_min_v < args.policy_cruise_v:
        raise ValueError(
            "policy_min_v must be non-negative and below policy_cruise_v")
    cfg = load_default_config(args.config_file)
    cfg.environment.num_envs = 1
    cfg.environment.camera_profile = (
        "diff_wheelbot_training" if args.sensor_profile == "training" else "dingo"
    )
    cfg.environment.control_dt = args.control_dt
    env, controller, scene_name = create_environment(
        cfg, scene_index=args.scene_index, device=args.device, seed=args.seed
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    speed_head_input = checkpoint_args.get("speed_head_input")
    if speed_head_input is None:
        # Legacy checkpoints used a hidden_dim-wide speed-head input. Infer
        # this before constructing the model so old evaluations remain valid.
        speed_weight = state_dict.get("speed_head.0.weight")
        hidden_dim = int(checkpoint_args.get("hidden_dim", 192))
        speed_head_input = (
            "hidden"
            if speed_weight is not None and speed_weight.shape[1] == hidden_dim
            else "fusion"
        )

    model_kwargs = dict(
        dim_obs=6,
        num_waypoints=int(checkpoint_args.get("num_waypoints", 3)),
        hidden_dim=int(checkpoint_args.get("hidden_dim", 192)),
        input_w=int(checkpoint_args.get("env_width", 64)) // 2,
        input_h=int(checkpoint_args.get("env_height", 48)) // 2,
        max_forward_step=float(
            checkpoint_args.get("max_forward_step", 1.5)
        ),
        max_lateral_step=float(
            checkpoint_args.get("max_lateral_step", 1.0)
        ),
        max_speed=args.policy_max_v,
        initial_desired_speed=args.policy_cruise_v,
        min_desired_speed=args.policy_min_v,
        hidden_decay=float(
            checkpoint_args.get(
                "hidden_decay",
                1.0,
            )
        ),
        speed_head_input=speed_head_input,
        goal_stop_distance=float(
            checkpoint_args.get("goal_stop_distance", 0.5)
        ),
        goal_slow_distance=float(
            checkpoint_args.get("goal_slow_distance", 2.0)
        ),
        direct_action=False,
    )
    # Model revisions use the same depth/goal/waypoint interface but do not
    # all expose newer constructor options (hidden decay, feed-forward or
    # distance speed modes). Pass only options supported by the selected repo.
    model_parameters = inspect.signature(Model.__init__).parameters
    model_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in model_parameters
    }
    model = Model(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    perception_safety_enabled = bool(
        checkpoint_args.get("mpc_perception_safety", True)
    )
    mpc_kwargs = dict(
        num_waypoints=int(checkpoint_args.get("num_waypoints", 3)),
        horizon=int(checkpoint_args.get("mpc_horizon", args.mpc_horizon)),
        control_lookahead=int(
            checkpoint_args.get(
                "mpc_control_lookahead", args.mpc_lookahead
            )
        ),
        max_v=args.max_v,
        max_omega=args.max_omega,
        max_acc_v=float(
            checkpoint_args.get("mpc_max_acc_v", args.mpc_max_acc_v)
        ),
        max_acc_omega=float(
            checkpoint_args.get(
                "mpc_max_acc_omega", args.mpc_max_acc_omega
            )
        ),
        max_lateral_acc=float(
            checkpoint_args.get(
                "mpc_max_lateral_acc", args.mpc_max_lateral_acc
            )
        ),
        track_weight=float(
            checkpoint_args.get("mpc_track_weight", args.mpc_track_weight)
        ),
        smooth_weight=float(
            checkpoint_args.get("mpc_smooth_weight", args.mpc_smooth_weight)
        ),
        initial_velocity_weight=float(
            checkpoint_args.get(
                "mpc_initial_velocity_weight",
                args.mpc_initial_velocity_weight,
            )
        ),
        perception_safety_enabled=perception_safety_enabled,
        obstacle_safety_clearance=float(
            checkpoint_args.get(
                "mpc_obstacle_clearance",
                args.mpc_obstacle_safety_clearance,
            )
        ),
        obstacle_temperature=float(
            checkpoint_args.get(
                "mpc_obstacle_temperature", args.mpc_obstacle_temperature
            )
        ),
        obstacle_weight=args.mpc_obstacle_weight,
        obstacle_refine_steps=int(
            checkpoint_args.get(
                "mpc_obstacle_refine_steps",
                args.mpc_obstacle_refine_steps,
            )
        ),
        obstacle_step_size=args.mpc_obstacle_step_size,
        intervention_weight=args.mpc_intervention_weight,
        obstacle_slowdown_gain=args.mpc_obstacle_slowdown_gain,
        obstacle_min_speed_scale=args.mpc_obstacle_min_speed_scale,
        ttc_clearance_inflation=args.mpc_ttc_clearance_inflation,
    )
    if not checkpoint_args:
        mpc_kwargs.update(
            emergency_max_acc_omega=args.mpc_emergency_max_acc_omega,
            emergency_smooth_weight=args.mpc_emergency_smooth_weight,
        )
    mpc = DifferentiableWaypointMPC(**mpc_kwargs).to(device).eval()

    def unpack_step_observation(step_output):
        """Return evaluator observations and done flags across Gym APIs."""
        if len(step_output) == 5:
            raw_obs, _, terminated, truncated, step_infos = step_output
            step_obs = step_infos.get("observations", raw_obs)
            step_dones = terminated | truncated
        else:
            raw_obs, _, step_dones, step_infos = step_output
            step_obs = step_infos.get("observations", raw_obs)
        return step_obs, step_dones, step_infos

    def contact_force_components(contact_sensor):
        """Return per-environment total, horizontal, and vertical contact forces."""
        force_xyz = contact_sensor.data.net_forces_w
        total = torch.linalg.vector_norm(force_xyz, dim=-1).amax(dim=1)
        horizontal = torch.linalg.vector_norm(force_xyz[..., :2], dim=-1).amax(dim=1)
        vertical = force_xyz[..., 2].abs().amax(dim=1)
        return total, horizontal, vertical

    def settle_spawn(step_obs, sample_index):
        """Settle spawn impulses at zero command before policy time starts."""
        clear_steps = 0
        max_total_force = 0.0
        max_horizontal_force = 0.0
        max_vertical_force = 0.0
        zero_command = np.zeros((1, 2), dtype=np.float32)
        for settle_step in range(1, args.collision_settle_max_steps + 1):
            wheel_action = controller.forward_batch(
                step_obs["policy"], zero_command
            ).to(device)
            step_obs, settle_dones, _ = unpack_step_observation(
                env.step(wheel_action)
            )
            if bool(settle_dones[0].item()):
                raise RuntimeError(
                    "Episode terminated during zero-command spawn settling "
                    f"for sample {sample_index} at settle step {settle_step}."
                )

            contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
            total_force, horizontal_force, vertical_force = contact_force_components(
                contact_sensor
            )
            total_force_value = float(total_force[0].detach().cpu())
            horizontal_force_value = float(horizontal_force[0].detach().cpu())
            vertical_force_value = float(vertical_force[0].detach().cpu())
            max_total_force = max(max_total_force, total_force_value)
            max_horizontal_force = max(
                max_horizontal_force, horizontal_force_value
            )
            max_vertical_force = max(max_vertical_force, vertical_force_value)
            if horizontal_force_value > args.contact_force_threshold:
                clear_steps = 0
            else:
                clear_steps += 1
            if clear_steps >= args.collision_settle_clear_steps:
                print(
                    "[collision_settle] "
                    f"sample={sample_index}, steps={settle_step}, "
                    f"clear_steps={clear_steps}, "
                    f"max_total={max_total_force:.3f}, "
                    f"max_horizontal={max_horizontal_force:.3f}, "
                    f"max_vertical={max_vertical_force:.3f}"
                )
                return step_obs

        raise RuntimeError(
            "Spawn contact did not clear while the robot was held at zero "
            f"command for sample {sample_index}: max_steps="
            f"{args.collision_settle_max_steps}, "
            f"max_total={max_total_force:.3f}, "
            f"max_horizontal={max_horizontal_force:.3f}, "
            f"max_vertical={max_vertical_force:.3f}, "
            f"threshold={args.contact_force_threshold:.3f}. The episode is "
            "rejected instead of hiding a possible initial collision."
        )

    reset_output = env.reset()
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        raw_obs, infos = reset_output
        obs = infos.get("observations", raw_obs)
    else:
        obs, infos = reset_output, {}
    initial_sample_idx = int(env.unwrapped._sample_idx[0].item())
    obs = settle_spawn(obs, initial_sample_idx)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dataset_dir = getattr(cfg.environment, "dataset_dir", None)
    if dataset_dir:
        scene_split = Path(dataset_dir).name
        scene_type = str(getattr(cfg.environment, "scene_type", scene_split))
        benchmark_dir = f"wheeled_internscene_{scene_type}"
    else:
        scene_split = Path(cfg.environment.scene_dir).name
        benchmark_dir = f"wheeled_{scene_split}"
    prealign_tag = "-prealign" if args.prealign_goal else ""
    if args.prealign_goal and args.prealign_safe_clearance > 0.0:
        prealign_tag += f"-safe{args.prealign_safe_clearance:g}"
    hidden_reset_tag = (
        f"-hreset{args.hidden_reset_interval}"
        if args.hidden_reset_interval > 0 else ""
    )
    recovery_tag = "-recovery" if args.stuck_recovery else ""
    generated_run_name = (
        f"{args.run_label}-{args.sensor_profile}-v{args.max_v:g}-w{args.max_omega:g}"
        f"{prealign_tag}{hidden_reset_tag}{recovery_tag}_{timestamp}"
    )
    run_dir_name = args.campaign_id or generated_run_name
    output_dir = (
        Path(cfg.run_root_dir) / benchmark_dir / run_dir_name
        / scene_split / scene_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = int(env.unwrapped._sample_idx[0].item())
    video_fps = max(1, int(round(1.0 / float(env.unwrapped.step_dt))))
    writer = (
        imageio.get_writer(
            str(output_dir / f"fps_{sample_idx}.mp4"), fps=video_fps
        )
        if args.record_video else None
    )
    initial_distance = float(
        torch.linalg.vector_norm(obs["goal_pose"][0, :2]).detach().cpu()
    )
    trajectory_length = 0.0
    completed = set()
    metrics = []
    hidden = None
    previous_depth_clearance = None
    goal_latched = torch.zeros(1, dtype=torch.bool, device=device)
    collision_latched = False
    collision_force_consecutive_count = 0
    collision_armed = True
    collision_clear_steps = args.collision_settle_clear_steps
    prealign_active = args.prealign_goal
    prealign_stable_count = 0
    prealign_step_count = 0
    # Zero means normal goal alignment.  +/-1 latches the selected obstacle
    # search direction so noisy left/right depth estimates cannot oscillate it.
    prealign_search_direction = 0
    terminal_approach_active = False
    policy_hidden_steps = 0
    stuck_low_speed_steps = 0
    stuck_window_start_distance = None
    recovery_phase = "idle"
    recovery_direction = 0
    recovery_previous_direction = 0
    recovery_previous_yaw = None
    recovery_rotated_angle = 0.0
    recovery_stable_count = 0
    recovery_sweep_count = 0
    recovery_cooldown_steps = 0
    recovery_trigger_count = 0
    recovery_success_count = 0
    recovery_blocked_count = 0
    prealign_tolerance = math.radians(args.prealign_tolerance_deg)
    step_count = 0
    episode_step = 0
    step_dt = float(env.unwrapped.step_dt)
    previous_actual_v = None
    linear_accel_sum_sq = 0.0
    linear_accel_sample_count = 0
    telemetry_file = open(output_dir / "telemetry.csv", "w", newline="", encoding="utf-8")
    telemetry_fields = (
        "step", "time_s", "episode_idx",
        "goal_distance", "goal_latched",
        "predicted_desired_speed", "predicted_speed_ratio", "speed_floor_hit",
        "mpc_desired_speed",
        "policy_speed_input", "real_current_speed",
        "wp1_x", "wp1_y", "wp2_x", "wp2_y", "wp3_x", "wp3_y",
        "wp1_dist", "wp2_dist", "wp3_dist",
        "hidden_norm", "hidden_mean", "hidden_std",
        "hidden_rms", "hidden_max_abs", "hidden_saturation_fraction",
        "command_v", "command_omega",
        "actual_v", "actual_omega",
        "emergency_risk", "perception_risk", "predicted_min_clearance",
        "contact_force", "contact_force_horizontal", "contact_force_vertical",
        "raw_collision", "collision", "collision_armed",
        "collision_clear_steps", "collision_force_consecutive_count",
        "control_phase", "goal_heading_error",
        "prealign_front_clearance", "prealign_search_direction",
        "terminal_approach_active", "terminal_front_clearance",
        "terminal_required_clearance",
        "stuck_low_speed_steps", "recovery_phase", "recovery_direction",
        "recovery_front_clearance", "recovery_left_score",
        "recovery_right_score", "recovery_rotated_deg",
        "recovery_cooldown_steps", "recovery_trigger_count",
        "recovery_success_count", "recovery_blocked_count",
        "hidden_reset",
        "stateless_policy",
        "hidden_decay",
    )
    telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry_fields)
    telemetry_writer.writeheader()

    print(f"[diff-wheelbot] checkpoint={args.checkpoint}")
    print(f"[diff-wheelbot] scene={scene_name}, sample={sample_idx}")
    speed_tag = (
        f"fixed@{args.fixed_speed:g}" if args.speed_mode == "fixed"
        else "learned"
    )
    stateless_tag = "stateless" if args.stateless_policy else "recurrent"
    speed_override_tag = (
        f"policy_speed_override={args.policy_speed_input_override:.3f}"
        if args.policy_speed_input_override is not None else "policy_speed=real"
    )
    print(
        f"[diff-wheelbot] sensor={args.sensor_profile}, dt={step_dt:.4f}s, "
        f"policy_max_v={args.policy_max_v:.3f}, execution_max_v={args.max_v:.3f}, "
        f"policy_min_v={args.policy_min_v:.3f}, max_omega={args.max_omega:.3f}, "
        f"goal_state_max_distance={args.goal_state_max_distance:.3f}, "
        f"fov_half_tan={args.fov_x_half_tan:.4f}, "
        f"seed={args.seed}, "
        f"hidden_reset_interval={args.hidden_reset_interval}, "
        f"collision_settle={args.collision_settle_clear_steps}/"
        f"{args.collision_settle_max_steps}, "
        f"collision_horizontal_threshold={args.contact_force_threshold:.3f}, "
        f"collision_confirm_steps={args.collision_force_consecutive_steps}, "
        f"collision_peak_threshold={args.collision_peak_force_threshold:.3f}, "
        f"video_front={args.video_front_view}, "
        f"speed_mode={speed_tag}, "
        f"gru_mode={stateless_tag}, stuck_recovery={int(args.stuck_recovery)}, "
        f"{speed_override_tag}"
    )

    try:
        while simulation_app.is_running():
            robot = env.unwrapped.scene["robot"]
            depth = obs["raw_depth"].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            depth = torch.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.2)
            depth = F.interpolate(depth, size=(48, 64), mode="area").clamp(0.2, 10.0)
            depth_input = F.max_pool2d(3.0 / depth - 0.6, 2, 2)

            if perception_safety_enabled:
                obstacle_points, obstacle_mask = depth_to_local_obstacle_points(
                    depth,
                    fov_x_half_tan=args.fov_x_half_tan,
                    num_points=int(
                        checkpoint_args.get(
                            "obstacle_num_points", args.mpc_obstacle_points
                        )
                    ),
                    height_fraction=float(
                        checkpoint_args.get(
                            "obstacle_height_fraction",
                            args.mpc_obstacle_height_fraction,
                        )
                    ),
                    depth_quantile=float(
                        checkpoint_args.get(
                            "obstacle_depth_quantile",
                            args.mpc_obstacle_depth_quantile,
                        )
                    ),
                    min_range=float(
                        checkpoint_args.get("obstacle_min_range", 0.2)
                    ),
                    max_range=float(
                        checkpoint_args.get(
                            "obstacle_max_range", args.mpc_obstacle_max_range
                        )
                    ),
                )
                emergency_risk, previous_depth_clearance = estimate_emergency_risk(
                    depth,
                    previous_depth_clearance,
                    step_dt,
                    robot_radius=args.robot_radius,
                    emergency_distance=float(
                        checkpoint_args.get(
                            "emergency_distance", args.emergency_distance
                        )
                    ),
                    emergency_ttc=float(
                        checkpoint_args.get("emergency_ttc", args.emergency_ttc)
                    ),
                    depth_quantile=args.emergency_depth_quantile,
                    depth_height_fraction=args.emergency_depth_height_fraction,
                )
                mpc_emergency_risk = emergency_risk
            else:
                obstacle_points = None
                obstacle_mask = None
                emergency_risk = torch.zeros(
                    (depth.shape[0], 1), dtype=depth.dtype, device=depth.device
                )
                mpc_emergency_risk = None

            goal_local = obs["goal_pose"][:, :2].to(device)
            distance = torch.linalg.vector_norm(goal_local, dim=1)
            goal_state_max_distance = float(args.goal_state_max_distance)
            if goal_state_max_distance <= 0.0:
                raise ValueError("--goal_state_max_distance must be positive")
            scale = torch.where(
                distance > goal_state_max_distance,
                goal_state_max_distance / distance.clamp_min(1e-6),
                torch.ones_like(distance),
            )
            goal_local = goal_local * scale[:, None]
            distance_scaled = distance * scale
            yaw = yaw_from_wxyz(robot.data.root_quat_w)
            current_speed = robot.data.root_lin_vel_b[:, 0:1].clamp(
                0.0, args.policy_max_v
            )
            actual_planar_speed = torch.linalg.vector_norm(
                robot.data.root_lin_vel_b[:, :2], dim=1
            )
            current_omega = robot.data.root_ang_vel_b[:, 2:3].clamp(
                -args.max_omega, args.max_omega
            )

            # ---- Policy speed input override (diagnostic only) ----
            if args.policy_speed_input_override is None:
                policy_speed_input = current_speed[:, 0]
            else:
                policy_speed_input = torch.full_like(
                    current_speed[:, 0],
                    args.policy_speed_input_override,
                )

            # The speed input is normalised by the policy speed cap so the
            # state is invariant to the configured max speed. Keep this
            # identical to the training-side state construction
            # (v_obs / max_speed); policy_max_v is aligned from the
            # checkpoint's max_speed.
            state = torch.stack(
                (
                    goal_local[:, 0] / goal_state_max_distance,
                    goal_local[:, 1] / goal_state_max_distance,
                    torch.cos(yaw),
                    torch.sin(yaw),
                    distance_scaled / goal_state_max_distance,
                    policy_speed_input / args.policy_max_v,
                ),
                dim=1,
            )

            if (
                args.terminal_approach
                and not terminal_approach_active
                and bool((distance[0] <= args.terminal_enter_distance).detach().cpu())
            ):
                terminal_approach_active = True
                prealign_active = False
                prealign_stable_count = 0
                prealign_search_direction = 0
                hidden = None

            with torch.no_grad():
                hidden_reset_now = False
                if (
                    not prealign_active
                    and args.hidden_reset_interval > 0
                    and policy_hidden_steps >= args.hidden_reset_interval
                ):
                    hidden = None
                    policy_hidden_steps = 0
                    hidden_reset_now = True

                if terminal_approach_active:
                    # Terminal docking is deliberately independent of the
                    # learned policy and MPC.  This also prevents recurrent
                    # state evolution after terminal-mode latching.
                    waypoints = torch.zeros(
                        (
                            depth.shape[0],
                            int(checkpoint_args.get("num_waypoints", 3)),
                            2,
                        ),
                        dtype=depth.dtype, device=depth.device,
                    )
                    predicted_desired_speed = torch.zeros(
                        (depth.shape[0], 1),
                        dtype=depth.dtype, device=depth.device,
                    )
                    hidden_out = None
                    hidden = None
                else:
                    # ---- Stateless policy: don't carry hidden across timesteps ----
                    recovery_was_active = recovery_phase != "idle"
                    if args.stateless_policy:
                        hidden_in = None
                    else:
                        hidden_in = hidden

                    waypoints, predicted_desired_speed, hidden_out = model(
                        depth_input, state, hidden_in,
                    )

                    if args.stateless_policy:
                        hidden = None
                    elif not recovery_was_active:
                        hidden = hidden_out
                    # While recovery commands are overriding the policy, keep
                    # the recurrent state frozen. The shadow forward pass is
                    # used only to detect a renewed forward intention.

                # ---- Speed-reference selection ----
                if args.speed_mode == "fixed":
                    desired_speed = torch.full_like(
                        predicted_desired_speed, args.fixed_speed,
                    )
                else:
                    desired_speed = predicted_desired_speed

                # ---- Waypoint diagnostics ----
                waypoint_dist = torch.linalg.vector_norm(waypoints, dim=-1)

                # ---- GRU hidden diagnostics ----
                if hidden_out is not None:
                    hidden_norm = torch.linalg.vector_norm(hidden_out, dim=1)
                    hidden_mean = hidden_out.mean(dim=1)
                    hidden_std = hidden_out.std(dim=1)
                    hidden_rms = hidden_out.pow(2).mean(dim=1).sqrt()
                    hidden_max_abs = hidden_out.abs().amax(dim=1)
                    hidden_saturation_fraction = (
                        hidden_out.abs() > 0.95
                    ).float().mean(dim=1)
                else:
                    hidden_norm = torch.zeros(1, device=device)
                    hidden_mean = torch.zeros(1, device=device)
                    hidden_std = torch.zeros(1, device=device)
                    hidden_rms = torch.zeros(1, device=device)
                    hidden_max_abs = torch.zeros(1, device=device)
                    hidden_saturation_fraction = torch.zeros(1, device=device)

                # ---- Speed ratio / floor-hit diagnostics ----
                speed_ratio = predicted_desired_speed / max(args.policy_max_v, 1e-6)
                speed_floor_hit = (
                    predicted_desired_speed <= args.policy_min_v + 0.02
                )
                if terminal_approach_active:
                    command = torch.zeros(
                        (depth.shape[0], 2),
                        dtype=depth.dtype, device=depth.device,
                    )
                else:
                    command, _ = mpc(
                        waypoints,
                        desired_speed,
                        current_speed,
                        step_dt,
                        current_omega=current_omega,
                        emergency_risk=mpc_emergency_risk,
                        obstacle_points=obstacle_points,
                        obstacle_mask=obstacle_mask,
                        obstacle_velocity=(
                            None
                            if obstacle_points is None
                            else torch.zeros_like(obstacle_points)
                        ),
                    )
                goal_heading_error = torch.atan2(goal_local[:, 1], goal_local[:, 0])
                control_phase = "policy"
                prealign_front_clearance = torch.full(
                    (depth.shape[0],), float("nan"),
                    dtype=depth.dtype, device=depth.device,
                )
                recovery_front_clearance = torch.full_like(
                    prealign_front_clearance, float("nan")
                )
                recovery_left_score = torch.full_like(
                    prealign_front_clearance, float("nan")
                )
                recovery_right_score = torch.full_like(
                    prealign_front_clearance, float("nan")
                )
                if prealign_active:
                    # Do not carry policy history accumulated while its command is
                    # being ignored.  The first policy step therefore starts with
                    # the same clean GRU state as it did during training rollouts.
                    hidden = None
                    control_phase = "prealign"
                    prealign_step_count += 1
                    timed_out = prealign_step_count * step_dt >= args.prealign_timeout

                    safe_alignment_enabled = args.prealign_safe_clearance > 0.0
                    if safe_alignment_enabled:
                        align_points, align_mask = depth_to_local_obstacle_points(
                            depth,
                            fov_x_half_tan=args.fov_x_half_tan,
                            num_points=args.prealign_obstacle_points,
                            height_fraction=args.prealign_depth_height_fraction,
                            depth_quantile=args.prealign_depth_quantile,
                            min_range=0.2,
                            max_range=args.mpc_obstacle_max_range,
                        )
                        align_x = align_points[..., 0]
                        align_y = align_points[..., 1]
                        in_corridor = (
                            align_mask
                            & (align_y.abs() <= args.prealign_corridor_half_width)
                        )
                        prealign_front_clearance = torch.where(
                            in_corridor,
                            align_x,
                            torch.full_like(align_x, float("inf")),
                        ).amin(dim=1)
                        prealign_front_clearance = torch.where(
                            torch.isfinite(prealign_front_clearance),
                            prealign_front_clearance,
                            torch.full_like(
                                prealign_front_clearance,
                                args.mpc_obstacle_max_range,
                            ),
                        )

                        # Invalid sectors normally mean no return inside the
                        # sensing range, which is useful open-space evidence.
                        sector_clearance = torch.where(
                            align_mask,
                            align_x,
                            torch.full_like(align_x, args.mpc_obstacle_max_range),
                        )
                        left_open = sector_clearance.masked_fill(
                            align_y <= 0.0, 0.0
                        ).mean(dim=1)
                        right_open = sector_clearance.masked_fill(
                            align_y >= 0.0, 0.0
                        ).mean(dim=1)
                    else:
                        left_open = torch.zeros_like(distance)
                        right_open = torch.zeros_like(distance)

                    goal_aligned = bool(
                        (goal_heading_error[0].abs() <= prealign_tolerance)
                        .detach().cpu()
                    )
                    omega_stable = bool(
                        (current_omega[0, 0].abs() <= args.prealign_omega_tolerance)
                        .detach().cpu()
                    )
                    release_clearance = args.prealign_safe_clearance
                    if prealign_search_direction != 0:
                        release_clearance += args.prealign_clearance_hysteresis
                    front_safe = (
                        not safe_alignment_enabled
                        or bool(
                            (prealign_front_clearance[0] >= release_clearance)
                            .detach().cpu()
                        )
                    )

                    if prealign_search_direction == 0 and goal_aligned and not front_safe:
                        prealign_search_direction = (
                            1
                            if float(left_open[0].detach().cpu())
                            >= float(right_open[0].detach().cpu())
                            else -1
                        )
                        prealign_stable_count = 0

                    if prealign_search_direction == 0:
                        aligned = goal_aligned and omega_stable and front_safe
                        prealign_stable_count = (
                            prealign_stable_count + 1 if aligned else 0
                        )
                        if prealign_stable_count >= args.prealign_stable_steps:
                            prealign_active = False
                            command.zero_()
                            control_phase = "prealign_complete"
                        elif timed_out and front_safe:
                            prealign_active = False
                            command.zero_()
                            control_phase = "prealign_timeout_safe"
                        else:
                            turn_rate = (
                                args.prealign_kp * goal_heading_error
                            ).clamp(
                                -args.prealign_max_omega,
                                args.prealign_max_omega,
                            )
                            command = torch.stack(
                                (torch.zeros_like(turn_rate), turn_rate), dim=1
                            )
                    else:
                        # Once a free ray is found, brake rotation and require
                        # several stable frames before handing over to policy.
                        # If inertia carries the camera back into blockage,
                        # resume the same latched search direction.
                        if front_safe:
                            command.zero_()
                            prealign_stable_count = (
                                prealign_stable_count + 1 if omega_stable else 0
                            )
                            control_phase = "prealign_search_settle"
                            if prealign_stable_count >= args.prealign_stable_steps:
                                prealign_active = False
                                control_phase = "prealign_search_complete"
                        else:
                            prealign_stable_count = 0
                            turn_rate = torch.full_like(
                                goal_heading_error,
                                prealign_search_direction * args.prealign_search_omega,
                            )
                            command = torch.stack(
                                (torch.zeros_like(turn_rate), turn_rate), dim=1
                            )
                            control_phase = (
                                "prealign_search_left"
                                if prealign_search_direction > 0
                                else "prealign_search_right"
                            )
                        if timed_out and not front_safe:
                            # Never turn a pre-alignment timeout into a blind
                            # launch.  Remaining stopped is safer and makes an
                            # invalid/fully enclosed start explicit in telemetry.
                            command.zero_()
                            control_phase = "prealign_blocked_timeout"

                if args.stuck_recovery and not terminal_approach_active:
                    confirm_steps = max(
                        1, int(math.ceil(args.recovery_confirm_time / step_dt))
                    )
                    cooldown_total_steps = max(
                        0, int(math.ceil(args.recovery_cooldown / step_dt))
                    )

                    if recovery_phase == "idle" and recovery_cooldown_steps > 0:
                        recovery_cooldown_steps -= 1

                    # Only normal policy control may accumulate stuck evidence.
                    # This excludes intentional stopping during pre-alignment,
                    # terminal docking, cooldown transitions, and recovery.
                    if (
                        recovery_phase == "idle"
                        and control_phase == "policy"
                        and recovery_cooldown_steps == 0
                    ):
                        low_command = bool(
                            (command[0, 0] <= args.recovery_command_v_threshold)
                            .detach().cpu()
                        )
                        low_actual = bool(
                            (actual_planar_speed[0] <= args.recovery_actual_v_threshold)
                            .detach().cpu()
                        )
                        goal_far = bool(
                            (distance[0] >= args.recovery_min_goal_distance)
                            .detach().cpu()
                        )
                        if low_command and low_actual and goal_far and not collision_latched:
                            if stuck_low_speed_steps == 0:
                                stuck_window_start_distance = float(
                                    distance[0].detach().cpu()
                                )
                            stuck_low_speed_steps += 1
                        else:
                            stuck_low_speed_steps = 0
                            stuck_window_start_distance = None

                        if stuck_low_speed_steps >= confirm_steps:
                            current_goal_distance = float(distance[0].detach().cpu())
                            goal_progress = max(
                                0.0,
                                float(stuck_window_start_distance)
                                - current_goal_distance,
                            )
                            if goal_progress <= args.recovery_progress_threshold:
                                recovery_phase = "rotate"
                                recovery_trigger_count += 1
                                recovery_previous_yaw = float(yaw[0].detach().cpu())
                                recovery_rotated_angle = 0.0
                                recovery_stable_count = 0
                                recovery_sweep_count = 1
                            else:
                                stuck_low_speed_steps = 0
                                stuck_window_start_distance = current_goal_distance

                    if recovery_phase != "idle":
                        recovery_points, recovery_mask = depth_to_local_obstacle_points(
                            depth,
                            fov_x_half_tan=args.fov_x_half_tan,
                            num_points=args.prealign_obstacle_points,
                            height_fraction=args.prealign_depth_height_fraction,
                            depth_quantile=args.prealign_depth_quantile,
                            min_range=0.2,
                            max_range=args.mpc_obstacle_max_range,
                        )
                        recovery_x = recovery_points[..., 0]
                        recovery_y = recovery_points[..., 1]
                        recovery_corridor = (
                            recovery_mask
                            & (recovery_y.abs() <= args.recovery_corridor_half_width)
                        )
                        recovery_front_clearance = torch.where(
                            recovery_corridor,
                            recovery_x,
                            torch.full_like(recovery_x, float("inf")),
                        ).amin(dim=1)
                        recovery_front_clearance = torch.where(
                            torch.isfinite(recovery_front_clearance),
                            recovery_front_clearance,
                            torch.full_like(
                                recovery_front_clearance,
                                args.mpc_obstacle_max_range,
                            ),
                        )

                        # A low depth quantile is a robust estimate of usable
                        # sector clearance: less noisy than the minimum and less
                        # optimistic than the mean in a narrow passage.
                        for batch_idx in range(depth.shape[0]):
                            left_values = recovery_x[batch_idx][
                                recovery_mask[batch_idx]
                                & (recovery_y[batch_idx] > 0.0)
                            ]
                            right_values = recovery_x[batch_idx][
                                recovery_mask[batch_idx]
                                & (recovery_y[batch_idx] < 0.0)
                            ]
                            recovery_left_score[batch_idx] = (
                                torch.quantile(left_values, 0.20)
                                if left_values.numel() > 0
                                else args.mpc_obstacle_max_range
                            )
                            recovery_right_score[batch_idx] = (
                                torch.quantile(right_values, 0.20)
                                if right_values.numel() > 0
                                else args.mpc_obstacle_max_range
                            )

                        if recovery_direction == 0:
                            heading_value = float(goal_heading_error[0].detach().cpu())
                            left_value = float(recovery_left_score[0].detach().cpu())
                            right_value = float(recovery_right_score[0].detach().cpu())
                            if heading_value > 0.0:
                                left_value += args.recovery_goal_bias
                            elif heading_value < 0.0:
                                right_value += args.recovery_goal_bias
                            if abs(left_value - right_value) <= 1e-3:
                                recovery_direction = (
                                    -recovery_previous_direction
                                    if recovery_previous_direction != 0 else 1
                                )
                            else:
                                recovery_direction = 1 if left_value > right_value else -1
                            recovery_previous_direction = recovery_direction

                        current_yaw_value = float(yaw[0].detach().cpu())
                        if recovery_previous_yaw is None:
                            recovery_previous_yaw = current_yaw_value
                        yaw_delta = math.atan2(
                            math.sin(current_yaw_value - recovery_previous_yaw),
                            math.cos(current_yaw_value - recovery_previous_yaw),
                        )
                        recovery_rotated_angle += abs(yaw_delta)
                        recovery_previous_yaw = current_yaw_value

                        min_turn = math.radians(args.recovery_min_turn_deg)
                        max_turn = math.radians(args.recovery_max_turn_deg)
                        front_safe = bool(
                            (recovery_front_clearance[0] >= args.recovery_front_clearance)
                            .detach().cpu()
                        )
                        strongly_open = bool(
                            (recovery_front_clearance[0] >= args.recovery_strong_clearance)
                            .detach().cpu()
                        )
                        policy_wants_forward = bool(
                            (predicted_desired_speed[0, 0]
                             >= args.recovery_policy_v_threshold)
                            .detach().cpu()
                        )
                        release_candidate = (
                            recovery_rotated_angle >= min_turn
                            and front_safe
                            and (policy_wants_forward or strongly_open)
                        )

                        if recovery_phase == "rotate":
                            sweep_angle_limit = (
                                max_turn
                                if recovery_sweep_count <= 1
                                else 2.0 * max_turn
                            )
                            control_phase = (
                                "recovery_rotate_left"
                                if recovery_direction > 0
                                else "recovery_rotate_right"
                            )
                            turn_rate = torch.full_like(
                                goal_heading_error,
                                recovery_direction * args.recovery_turn_omega,
                            )
                            command = torch.stack(
                                (torch.zeros_like(turn_rate), turn_rate), dim=1
                            )
                            if release_candidate:
                                recovery_phase = "settle"
                                recovery_stable_count = 0
                                command.zero_()
                                control_phase = "recovery_settle"
                            if (
                                recovery_phase == "rotate"
                                and recovery_rotated_angle >= sweep_angle_limit
                            ):
                                if recovery_sweep_count < args.recovery_max_sweeps:
                                    recovery_direction *= -1
                                    recovery_previous_direction = recovery_direction
                                    recovery_previous_yaw = current_yaw_value
                                    recovery_rotated_angle = 0.0
                                    recovery_sweep_count += 1
                                else:
                                    recovery_phase = "blocked"
                                    recovery_blocked_count += 1
                                    command.zero_()
                                    control_phase = "recovery_blocked"
                        elif recovery_phase == "settle":
                            command.zero_()
                            control_phase = "recovery_settle"
                            omega_stable = bool(
                                (current_omega[0, 0].abs()
                                 <= args.recovery_omega_tolerance)
                                .detach().cpu()
                            )
                            recovery_stable_count = (
                                recovery_stable_count + 1
                                if release_candidate and omega_stable else 0
                            )
                            if not front_safe:
                                recovery_phase = "rotate"
                                recovery_stable_count = 0
                            elif recovery_stable_count >= args.recovery_stable_steps:
                                recovery_phase = "idle"
                                recovery_success_count += 1
                                recovery_direction = 0
                                recovery_previous_yaw = None
                                recovery_rotated_angle = 0.0
                                recovery_stable_count = 0
                                recovery_sweep_count = 0
                                recovery_cooldown_steps = cooldown_total_steps
                                stuck_low_speed_steps = 0
                                stuck_window_start_distance = None
                                hidden = None
                                hidden_reset_now = True
                                policy_hidden_steps = 0
                                command.zero_()
                                control_phase = "recovery_complete"
                        else:
                            command.zero_()
                            control_phase = "recovery_blocked"

                if control_phase == "policy":
                    policy_hidden_steps += 1
                else:
                    policy_hidden_steps = 0
                if args.open_loop_v is not None or args.open_loop_omega is not None:
                    command = torch.tensor(
                        [[args.open_loop_v or 0.0, args.open_loop_omega or 0.0]],
                        dtype=command.dtype,
                        device=command.device,
                    ).expand_as(command)

                terminal_front_clearance = torch.full(
                    (depth.shape[0],), float("nan"),
                    dtype=depth.dtype, device=depth.device,
                )
                terminal_required_clearance = torch.full(
                    (depth.shape[0],), float("nan"),
                    dtype=depth.dtype, device=depth.device,
                )
                if terminal_approach_active:
                    terminal_points, terminal_mask = depth_to_local_obstacle_points(
                        depth,
                        fov_x_half_tan=args.fov_x_half_tan,
                        num_points=args.prealign_obstacle_points,
                        height_fraction=args.prealign_depth_height_fraction,
                        depth_quantile=args.prealign_depth_quantile,
                        min_range=0.2,
                        max_range=args.mpc_obstacle_max_range,
                    )
                    terminal_x = terminal_points[..., 0]
                    terminal_y = terminal_points[..., 1]
                    terminal_corridor = (
                        terminal_mask
                        & (terminal_y.abs() <= args.terminal_corridor_half_width)
                    )
                    terminal_front_clearance = torch.where(
                        terminal_corridor,
                        terminal_x,
                        torch.full_like(terminal_x, float("inf")),
                    ).amin(dim=1)
                    terminal_front_clearance = torch.where(
                        torch.isfinite(terminal_front_clearance),
                        terminal_front_clearance,
                        torch.full_like(
                            terminal_front_clearance,
                            args.mpc_obstacle_max_range,
                        ),
                    )

                    terminal_omega = (
                        args.terminal_k_omega * goal_heading_error
                    ).clamp(-args.terminal_max_omega, args.terminal_max_omega)
                    terminal_target_distance = (
                        args.goal_stop_distance - args.terminal_goal_margin
                    )
                    terminal_v = (
                        args.terminal_k_v
                        * (distance - terminal_target_distance).clamp_min(0.0)
                    ).clamp_max(args.terminal_max_v)
                    terminal_v = terminal_v * torch.cos(
                        goal_heading_error
                    ).clamp_min(0.0)
                    heading_safe = (
                        goal_heading_error.abs()
                        <= math.radians(args.terminal_heading_stop_deg)
                    )
                    remaining_to_success = (
                        distance - args.goal_stop_distance
                    ).clamp_min(0.0)
                    terminal_required_clearance = (
                        remaining_to_success + args.terminal_safe_clearance
                    )
                    depth_safe = (
                        terminal_front_clearance >= terminal_required_clearance
                    )
                    terminal_v = torch.where(
                        heading_safe & depth_safe,
                        terminal_v,
                        torch.zeros_like(terminal_v),
                    )
                    command = torch.stack((terminal_v, terminal_omega), dim=1)
                    control_phase = (
                        "terminal_approach"
                        if bool(heading_safe[0].detach().cpu())
                        else "terminal_align"
                    )
                    policy_hidden_steps = 0
                goal_latched |= distance <= args.goal_stop_distance
                command = torch.where(goal_latched[:, None], torch.zeros_like(command), command)
                if bool(goal_latched[0].detach().cpu()):
                    control_phase = "terminal_complete"

                # ---- Console diagnostic every 30 policy steps ----
                if (
                    control_phase == "policy"
                    and policy_hidden_steps > 0
                    and policy_hidden_steps % 30 == 0
                ):
                    print(
                        "[diag] "
                        f"step={episode_step} "
                        f"real_v={float(current_speed[0, 0].detach().cpu()):.3f} "
                        f"policy_v={float(policy_speed_input[0].detach().cpu()):.3f} "
                        f"pred_v={float(predicted_desired_speed[0, 0].detach().cpu()):.3f} "
                        f"mpc_v={float(desired_speed[0, 0].detach().cpu()):.3f} "
                        f"cmd_v={float(command[0, 0].detach().cpu()):.3f} "
                        f"wp1={float(waypoint_dist[0, 0].detach().cpu()):.3f} "
                        f"wp3={float(waypoint_dist[0, 2].detach().cpu()):.3f} "
                        f"h_rms={float(hidden_rms[0].detach().cpu()):.3f} "
                        f"h_sat={float(hidden_saturation_fraction[0].detach().cpu()):.3f} "
                        f"stateless={int(args.stateless_policy)}"
                    )

            contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
            (
                contact_force,
                contact_force_horizontal,
                contact_force_vertical,
            ) = contact_force_components(contact_sensor)
            raw_collision_now = bool(
                (
                    contact_force_horizontal[0]
                    > args.contact_force_threshold
                ).detach().cpu()
            )
            if raw_collision_now:
                collision_force_consecutive_count += 1
            else:
                collision_force_consecutive_count = 0
            peak_collision_now = bool(
                (
                    contact_force_horizontal[0]
                    >= args.collision_peak_force_threshold
                ).detach().cpu()
            )
            collision_now = (
                collision_force_consecutive_count
                >= args.collision_force_consecutive_steps
            ) or peak_collision_now
            collision_latched = collision_latched or collision_now

            actual_v_value = float(current_speed[0, 0].detach().cpu())
            if previous_actual_v is not None:
                linear_accel = (actual_v_value - previous_actual_v) / step_dt
                linear_accel_sum_sq += linear_accel * linear_accel
                linear_accel_sample_count += 1
            previous_actual_v = actual_v_value
            telemetry_writer.writerow(
                {
                    "step": episode_step,
                    "time_s": episode_step * step_dt,
                    "episode_idx": sample_idx,
                    "goal_distance": float(distance[0].detach().cpu()),
                    "goal_latched": int(bool(goal_latched[0].detach().cpu())),
                    "predicted_desired_speed": float(predicted_desired_speed[0, 0].detach().cpu()),
                    "predicted_speed_ratio": float(speed_ratio[0, 0].detach().cpu()),
                    "speed_floor_hit": int(bool(speed_floor_hit[0, 0].detach().cpu())),
                    "mpc_desired_speed": float(desired_speed[0, 0].detach().cpu()),
                    "wp1_x": float(waypoints[0, 0, 0].detach().cpu()),
                    "wp1_y": float(waypoints[0, 0, 1].detach().cpu()),
                    "wp2_x": float(waypoints[0, 1, 0].detach().cpu()),
                    "wp2_y": float(waypoints[0, 1, 1].detach().cpu()),
                    "wp3_x": float(waypoints[0, 2, 0].detach().cpu()),
                    "wp3_y": float(waypoints[0, 2, 1].detach().cpu()),
                    "wp1_dist": float(waypoint_dist[0, 0].detach().cpu()),
                    "wp2_dist": float(waypoint_dist[0, 1].detach().cpu()),
                    "wp3_dist": float(waypoint_dist[0, 2].detach().cpu()),
                    "hidden_norm": float(hidden_norm[0].detach().cpu()),
                    "hidden_mean": float(hidden_mean[0].detach().cpu()),
                    "hidden_std": float(hidden_std[0].detach().cpu()),
                    "hidden_rms": float(hidden_rms[0].detach().cpu()),
                    "hidden_max_abs": float(hidden_max_abs[0].detach().cpu()),
                    "hidden_saturation_fraction": float(hidden_saturation_fraction[0].detach().cpu()),
                    "policy_speed_input": float(policy_speed_input[0].detach().cpu()),
                    "real_current_speed": float(current_speed[0, 0].detach().cpu()),
                    "command_v": float(command[0, 0].detach().cpu()),
                    "command_omega": float(command[0, 1].detach().cpu()),
                    "actual_v": actual_v_value,
                    "actual_omega": float(current_omega[0, 0].detach().cpu()),
                    "emergency_risk": float(emergency_risk[0, 0].detach().cpu()),
                    "perception_risk": (
                        0.0 if terminal_approach_active else
                        float(mpc.last_info["perception_risk"][0, 0].detach().cpu())
                    ),
                    "predicted_min_clearance": (
                        float("inf") if terminal_approach_active else
                        float(mpc.last_info["predicted_min_clearance"][0, 0].detach().cpu())
                    ),
                    "contact_force": float(contact_force[0].detach().cpu()),
                    "contact_force_horizontal": float(
                        contact_force_horizontal[0].detach().cpu()
                    ),
                    "contact_force_vertical": float(
                        contact_force_vertical[0].detach().cpu()
                    ),
                    "raw_collision": int(raw_collision_now),
                    "collision": int(collision_now),
                    "collision_armed": int(collision_armed),
                    "collision_clear_steps": collision_clear_steps,
                    "collision_force_consecutive_count": (
                        collision_force_consecutive_count
                    ),
                    "control_phase": control_phase,
                    "goal_heading_error": float(
                        goal_heading_error[0].detach().cpu()
                    ),
                    "prealign_front_clearance": float(
                        prealign_front_clearance[0].detach().cpu()
                    ),
                    "prealign_search_direction": prealign_search_direction,
                    "terminal_approach_active": int(terminal_approach_active),
                    "terminal_front_clearance": float(
                        terminal_front_clearance[0].detach().cpu()
                    ),
                    "terminal_required_clearance": float(
                        terminal_required_clearance[0].detach().cpu()
                    ),
                    "stuck_low_speed_steps": stuck_low_speed_steps,
                    "recovery_phase": recovery_phase,
                    "recovery_direction": recovery_direction,
                    "recovery_front_clearance": float(
                        recovery_front_clearance[0].detach().cpu()
                    ),
                    "recovery_left_score": float(
                        recovery_left_score[0].detach().cpu()
                    ),
                    "recovery_right_score": float(
                        recovery_right_score[0].detach().cpu()
                    ),
                    "recovery_rotated_deg": math.degrees(recovery_rotated_angle),
                    "recovery_cooldown_steps": recovery_cooldown_steps,
                    "recovery_trigger_count": recovery_trigger_count,
                    "recovery_success_count": recovery_success_count,
                    "recovery_blocked_count": recovery_blocked_count,
                    "hidden_reset": int(hidden_reset_now),
                    "stateless_policy": int(args.stateless_policy),
                    "hidden_decay": float(getattr(model, "hidden_decay", 1.0)),
                }
            )
            if episode_step % 100 == 0:
                telemetry_file.flush()

            planar_speed = torch.linalg.vector_norm(
                robot.data.root_lin_vel_w[:, :2], dim=1
            )
            trajectory_length += float(planar_speed[0].detach().cpu()) * step_dt
            wheel_action = controller.forward_batch(
                obs["policy"], command.detach().cpu().numpy()
            ).to(device)

            step_output = env.step(wheel_action)
            obs, dones, infos = unpack_step_observation(step_output)
            step_count += 1
            episode_step += 1

            if writer is not None:
                if args.video_front_view == "depth":
                    # Fixed-range metric-depth visualization: nearby obstacles
                    # are black and distant/invalid pixels are white.
                    # Nearest-neighbour upsampling exposes the exact 64x48
                    # policy measurement without inventing smooth detail.
                    front_depth = (
                        obs["raw_depth"][0, ..., 0].detach().cpu().numpy()
                    )
                    front_depth = np.nan_to_num(
                        front_depth,
                        nan=args.video_depth_far,
                        posinf=args.video_depth_far,
                        neginf=args.video_depth_near,
                    )
                    depth_normalized = (
                        np.clip(
                            front_depth,
                            args.video_depth_near,
                            args.video_depth_far,
                        )
                        - args.video_depth_near
                    ) / (args.video_depth_far - args.video_depth_near)
                    front_gray = np.round(
                        depth_normalized * 255.0
                    ).astype(np.uint8)
                    front_gray = cv2.resize(
                        front_gray,
                        (384, 384),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    rgb = np.repeat(front_gray[..., None], 3, axis=2)
                else:
                    rgb = cv2.resize(
                        obs["raw_rgb"][0].detach().cpu().numpy(), (384, 384)
                    )
                if "birdeye_rgb" in obs:
                    bird = cv2.resize(
                        obs["birdeye_rgb"][0].detach().cpu().numpy(), (384, 384)
                    )
                else:
                    bird = np.zeros_like(rgb)
                writer.append_data(np.concatenate((rgb, bird), axis=1))

            if bool(dones[0].item()):
                timeout = float(infos["time_outs"][0].item())
                success = 1.0 - timeout
                metrics.append(
                    {
                        "success": success,
                        "spl": (
                            initial_distance
                            / max(trajectory_length, initial_distance, 1e-8)
                        )
                        * success,
                        "distance": initial_distance,
                        "episode_idx": sample_idx,
                        "trajectory_length": trajectory_length,
                        "elapsed_time": episode_step * step_dt,
                        "collision": float(collision_latched),
                        "linear_accel_rms": math.sqrt(
                            linear_accel_sum_sq
                            / max(linear_accel_sample_count, 1)
                        ),
                    }
                )
                completed.add(sample_idx)
                if writer is not None:
                    writer.close()
                write_metrics(metrics, output_dir / "metric.csv")
                print(f"[diff-wheelbot] episode={sample_idx} success={success:.0f}")
                if len(completed) >= args.num_episodes:
                    break

                hidden = None
                previous_depth_clearance = None
                goal_latched.zero_()
                collision_latched = False
                collision_force_consecutive_count = 0
                collision_armed = True
                collision_clear_steps = args.collision_settle_clear_steps
                prealign_active = args.prealign_goal
                prealign_stable_count = 0
                prealign_step_count = 0
                prealign_search_direction = 0
                terminal_approach_active = False
                policy_hidden_steps = 0
                stuck_low_speed_steps = 0
                stuck_window_start_distance = None
                recovery_phase = "idle"
                recovery_direction = 0
                recovery_previous_direction = 0
                recovery_previous_yaw = None
                recovery_rotated_angle = 0.0
                recovery_stable_count = 0
                recovery_sweep_count = 0
                recovery_cooldown_steps = 0
                recovery_trigger_count = 0
                recovery_success_count = 0
                recovery_blocked_count = 0
                trajectory_length = 0.0
                episode_step = 0
                previous_actual_v = None
                linear_accel_sum_sq = 0.0
                linear_accel_sample_count = 0
                sample_idx = int(env.unwrapped._sample_idx[0].item())
                obs = settle_spawn(obs, sample_idx)
                initial_distance = float(
                    torch.linalg.vector_norm(obs["goal_pose"][0, :2]).detach().cpu()
                )
                writer = (
                    imageio.get_writer(
                        str(output_dir / f"fps_{sample_idx}.mp4"), fps=video_fps
                    )
                    if args.record_video else None
                )

            if args.max_steps is not None and step_count >= args.max_steps:
                print(f"[diff-wheelbot] reached --max_steps={args.max_steps}")
                break
    finally:
        telemetry_file.flush()
        telemetry_file.close()
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()

    print(f"[diff-wheelbot] output={output_dir}")


if __name__ == "__main__":
    main()
