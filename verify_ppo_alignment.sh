#!/usr/bin/env bash
# PPO baseline 与主模型对齐的冒烟检查：
#   1) 语法检查三个修改过的脚本
#   2) 用 configs/ppo_train.args 实际解析一遍参数
#   3) 校验关键对齐项取值，并打印课程表确认密度调度正确
# 用法: bash verify_ppo_alignment.sh
set -e
cd "$(dirname "$0")"

python3 -m py_compile train_ppo.py evaluate_benchmark.py compare_baselines.py
echo "py_compile OK"

python3 - <<'EOF'
import math
import sys

sys.argv = ['train_ppo.py', '@configs/ppo_train.args']
from train_ppo import parse_args

args = parse_args()

checks = [
    ('num_cyl', args.num_cyl, 27),
    ('num_balls', args.num_balls, 15),
    ('num_vox', args.num_vox, 10),
    ('batch_size', args.batch_size, 256),
    ('hidden_dim', args.hidden_dim, 192),
    ('grad_decay', args.grad_decay, 0.8),
    ('cyl_height_min', args.cyl_height_min, 0.5),
    ('cyl_height_max', args.cyl_height_max, 1.5),
    ('cruise_v', args.cruise_v, 1.6),
    ('goal_state_max_distance', args.goal_state_max_distance, 10.0),
    ('randomize_horizon', args.randomize_horizon, True),
    ('min_timesteps', args.min_timesteps, 135),
    ('max_timesteps', args.max_timesteps, 165),
    ('obstacle_curriculum', args.obstacle_curriculum, True),
    ('obstacle_curriculum_mode', args.obstacle_curriculum_mode, 'mixed'),
    ('seed', args.seed, 1),
    ('control_hz', args.control_hz, 15.0),
    ('dt_noise_std', args.dt_noise_std, 0.005),
    ('max_action_delay_steps', args.max_action_delay_steps, 1),
    ('robot_radius', args.robot_radius, 0.24),
    ('map_size', args.map_size, 14.0),
    ('protected_zone_radius', args.protected_zone_radius, 1.0),
]
failed = False
for name, actual, expected in checks:
    if actual != expected:
        failed = True
        print(f'MISMATCH: {name} = {actual} (expected {expected})')
    else:
        print(f'OK: {name} = {actual}')

total_updates = math.ceil(
    args.total_timesteps / (args.batch_size * args.rollout_steps))
print(f'total_updates = {total_updates}')

from train_mpc import obstacle_curriculum_state
for update in (0, 2999, 3000, 8999, 9000, total_updates - 1):
    phase, level, counts = obstacle_curriculum_state(args, update)
    print(f'update {update:4d}: phase={phase} counts={counts}')

if failed:
    sys.exit(1)
print('ALIGNMENT CHECK PASSED')
EOF

python3 - <<'EOF'
from argparse import Namespace

import torch

from train_ppo import compute_gae

args = Namespace(gamma=0.9, gae_lambda=1.0)
rollout = {
    'reward': torch.tensor([[1.0], [1.0]]),
    'old_value': torch.tensor([[2.0], [3.0]]),
    'done': torch.tensor([[False], [False]]),
    'valid': torch.tensor([[True], [True]]),
    'bootstrap_value': torch.tensor([4.0]),
}
_, returns = compute_gae(args, rollout)
torch.testing.assert_close(returns, torch.tensor([[5.14], [4.60]]))

terminal = dict(rollout)
terminal['done'] = torch.tensor([[False], [True]])
_, terminal_returns = compute_gae(args, terminal)
torch.testing.assert_close(terminal_returns, torch.tensor([[1.90], [1.0]]))
print('TIME-LIMIT BOOTSTRAP CHECK PASSED')
EOF

python3 - <<'EOF'
import sys

from model_mpc import Model
from ppo_model import PPOActorCritic
from train_mpc import parse_args as parse_main_args

sys.argv = [
    'train_mpc.py',
    '@configs/xnavdp_dense_corridor_gate_retention_v1.args',
]
main_args = parse_main_args()

sys.argv = ['train_ppo.py', '@configs/ppo_train.args']
from train_ppo import parse_args as parse_ppo_args
ppo_args = parse_ppo_args()

shared_config = (
    'batch_size', 'grad_decay', 'fov_x_half_tan', 'control_hz',
    'dt_noise_std', 'motor_rate_min', 'motor_rate_max',
    'max_action_delay_steps', 'exec_v_scale_std', 'exec_w_scale_std',
    'wheel_bias_std', 'min_timesteps', 'max_timesteps', 'hidden_dim', 'max_speed',
    'max_omega', 'goal_state_max_distance', 'map_size', 'num_cyl',
    'num_balls', 'num_vox', 'robot_radius', 'cyl_radius_min',
    'cyl_radius_max', 'cyl_height_min', 'cyl_height_max',
    'ball_radius_min', 'ball_radius_max', 'ball_radius_floor',
    'randomize_start_goal', 'protected_zone_radius', 'obstacle_layout',
    'obstacle_grid_jitter', 'obstacle_candidate_multiplier',
    'dynamic_obstacle_scene_prob', 'dynamic_obstacle_ratio',
    'dynamic_obstacle_speed_min', 'dynamic_obstacle_speed_max',
)
for name in shared_config:
    actual = getattr(ppo_args, name)
    expected = getattr(main_args, name)
    if actual != expected:
        raise AssertionError(f'{name}: PPO={actual!r}, main={expected!r}')

main_model = Model(
    dim_obs=6, hidden_dim=192, input_w=32, input_h=24,
    max_speed=2.0, initial_desired_speed=1.6,
)
ppo_model = PPOActorCritic(
    dim_obs=6, hidden_dim=192, input_w=32, input_h=24,
    max_v=2.0, max_omega=3.0, initial_speed=1.6,
)
main_shapes = {
    key: tuple(value.shape)
    for key, value in main_model.state_dict().items()
    if key.startswith(('conv_net.', 'fc_visual.', 'fc_state.', 'gru.'))
}
ppo_shapes = {
    key: tuple(value.shape)
    for key, value in ppo_model.state_dict().items()
    if key.startswith(('conv_net.', 'fc_visual.', 'fc_state.', 'gru.'))
}
if main_shapes != ppo_shapes:
    raise AssertionError('PPO encoder/GRU shapes do not match the main model')
print('MAIN/PPO CONFIG AND BACKBONE ALIGNMENT CHECK PASSED')
EOF
