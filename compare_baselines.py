"""Compare the waypoint+MPC main model against the PPO baseline.

Each checkpoint is evaluated in-process over the same benchmark scenes with
``evaluate_benchmark`` settings, then an aligned table is printed and saved.

Usage:
    python3 compare_baselines.py \
        --main save/xnavdp_dense_corridor_v2/seed1_.../checkpoint_mpc_5000.pth \
        --ppo save/ppo_baseline/seed0_.../checkpoint_ppo_final.pth \
        --benchmark all --episodes 256
"""

import argparse
import csv
import json
from pathlib import Path

import torch

import evaluate_benchmark as eb


TABLE_METRICS = (
    ('success_rate', '{:.3f}'),
    ('collision_rate', '{:.3f}'),
    ('timeout_rate', '{:.3f}'),
    ('spl', '{:.3f}'),
    ('time_to_goal_mean', '{:.2f}'),
    ('final_distance_mean', '{:.2f}'),
    ('task_min_clearance_min', '{:.2f}'),
    ('linear_speed_mean', '{:.2f}'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--main', action='append', default=None, dest='main_checkpoints',
        help='Waypoint+MPC checkpoint (repeatable)')
    parser.add_argument(
        '--ppo', action='append', default=None, dest='ppo_checkpoints',
        help='PPO baseline checkpoint (repeatable)')
    parser.add_argument(
        '--benchmark', choices=('all', *eb.BENCHMARKS), default='all')
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--output_prefix', default='save/comparison/benchmark',
        help='Prefix for the saved CSV/JSON comparison files')
    # Evaluation-only overrides forwarded to evaluate_benchmark.
    parser.add_argument('--success_radius', type=float, default=None)
    parser.add_argument('--goal_stop_distance', type=float, default=0.5)
    parser.add_argument('--safety_margin', type=float, default=0.3)
    parser.add_argument('--depth_noise_std', type=float, default=0.02)

    args = parser.parse_args()
    if not args.main_checkpoints and not args.ppo_checkpoints:
        parser.error('Provide at least one --main or --ppo checkpoint')
    for path in (args.main_checkpoints or []) + (args.ppo_checkpoints or []):
        if not Path(path).is_file():
            parser.error(f'Checkpoint not found: {path}')
    if args.episodes <= 0 or args.batch_size <= 0 or args.timesteps <= 0:
        parser.error(
            '--episodes, --batch_size and --timesteps must be positive')
    return args


def build_eval_args(args, checkpoint, policy_type):
    return argparse.Namespace(
        checkpoint=checkpoint,
        policy_type=policy_type,
        benchmark=args.benchmark,
        episodes=args.episodes,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        output=None,
        episode_output=None,
        success_radius=args.success_radius,
        goal_stop_distance=args.goal_stop_distance,
        safety_margin=args.safety_margin,
        depth_noise_std=args.depth_noise_std,
    )


def evaluate_one(args, checkpoint, policy_type):
    eval_args = build_eval_args(args, checkpoint, policy_type)
    device = torch.device(args.device)
    checkpoint_data = eb.load_checkpoint(eval_args, device)
    cfg = eb.resolve_config(eval_args, checkpoint_data)
    model = eb.load_model(cfg, checkpoint_data, device)
    results, _ = eb.evaluate_checkpoint(eval_args, model=model, cfg=cfg)
    return results


def default_label(checkpoint, policy_type, index):
    name = Path(checkpoint).stem
    return f'{policy_type}:{name}' if index else name


def print_table(all_results, benchmark_names):
    header_metrics = ' | '.join(name for name, _ in TABLE_METRICS)
    for benchmark in benchmark_names:
        print(f'\n=== {benchmark} ===')
        print(f'{"policy":<40} | {header_metrics}')
        print('-' * (43 + len(header_metrics)))
        for label, results in all_results:
            metrics = results.get(benchmark)
            if metrics is None:
                continue
            cells = []
            for name, fmt in TABLE_METRICS:
                value = metrics.get(name)
                cells.append(
                    'n/a' if value is None else fmt.format(value))
            print(f'{label:<40} | ' + ' | '.join(cells))


def main():
    args = parse_args()
    targets = []
    for index, checkpoint in enumerate(args.main_checkpoints or []):
        targets.append(
            (default_label(checkpoint, 'waypoint', index), checkpoint,
             'waypoint'))
    for index, checkpoint in enumerate(args.ppo_checkpoints or []):
        targets.append(
            (default_label(checkpoint, 'ppo', index), checkpoint, 'ppo'))

    all_results = []
    for label, checkpoint, policy_type in targets:
        print(f'\n>>> Evaluating {label} ({checkpoint})')
        results = evaluate_one(args, checkpoint, policy_type)
        all_results.append((label, results))
        for name, metrics in results.items():
            eb.print_summary_line(f'{label[:28]}@{name}', metrics)

    benchmark_names = (
        list(eb.DEFAULT_BENCHMARKS) if args.benchmark == 'all'
        else [args.benchmark])
    print_table(all_results, benchmark_names)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix('.csv')
    json_path = output_prefix.with_suffix('.json')

    fieldnames = ['policy', 'checkpoint', 'benchmark'] + [
        name for name, _ in TABLE_METRICS]
    checkpoint_by_label = {label: checkpoint for label, checkpoint, _ in targets}
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, results in all_results:
            checkpoint = checkpoint_by_label[label]
            for benchmark in benchmark_names:
                metrics = results.get(benchmark)
                if metrics is None:
                    continue
                row = {
                    'policy': label,
                    'checkpoint': checkpoint,
                    'benchmark': benchmark,
                }
                for name, _ in TABLE_METRICS:
                    row[name] = metrics.get(name)
                writer.writerow(row)
    json_path.write_text(
        json.dumps(
            {label: results for label, results in all_results}, indent=2),
        encoding='utf-8')
    print(f'\nWrote comparison table to {csv_path} and {json_path}')


if __name__ == '__main__':
    main()
