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
    ('grad_decay', args.grad_decay, 1.0),
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
for update in (0, 103, 104, 312, 313, total_updates - 1):
    phase, level, counts = obstacle_curriculum_state(args, update)
    print(f'update {update:4d}: phase={phase} counts={counts}')

if failed:
    sys.exit(1)
print('ALIGNMENT CHECK PASSED')
EOF
