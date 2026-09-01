#!/usr/bin/env python3
"""Summarize one Diff-Wheelbot InternScene evaluation campaign."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=20)
    parser.add_argument("--expected-episodes", type=int, default=100)
    return parser.parse_args()


def read_metric(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "success": float(row["success"]),
                        "spl": float(row["spl"]),
                        "collision": float(row.get("collision", 0.0)),
                        "trajectory_length": float(row.get("trajectory_length", 0.0)),
                        "elapsed_time": float(row.get("elapsed_time", 0.0)),
                        "linear_accel_rms": float(
                            row.get("linear_accel_rms", "nan") or "nan"
                        ),
                        "episode_idx": float(row.get("episode_idx", -1)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return mean(finite_values) if finite_values else float("nan")


def main() -> int:
    args = parse_args()
    root = args.campaign_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted(root.rglob("metric.csv"))

    scene_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, float]] = []
    for metric_path in metric_paths:
        rows = read_metric(metric_path)
        all_rows.extend(rows)
        episode_ids = [int(row["episode_idx"]) for row in rows]
        unique_episodes = len(set(episode_ids))
        scene_rows.append(
            {
                "scene": metric_path.parent.name,
                "episodes": len(rows),
                "unique_episodes": unique_episodes,
                "successes": sum(row["success"] >= 0.5 for row in rows),
                "sr": mean([row["success"] for row in rows]),
                "spl": mean([row["spl"] for row in rows]),
                "collision_rate": mean([row["collision"] for row in rows]),
                "collision_free_sr": mean(
                    [
                        float(row["success"] >= 0.5 and row["collision"] < 0.5)
                        for row in rows
                    ]
                ),
                "mean_linear_accel_rms": finite_mean(
                    [row["linear_accel_rms"] for row in rows]
                ),
                "mean_trajectory_length": mean(
                    [row["trajectory_length"] for row in rows]
                ),
                "mean_elapsed_time": mean([row["elapsed_time"] for row in rows]),
                "complete": int(
                    len(rows) == args.expected_episodes
                    and unique_episodes == args.expected_episodes
                ),
                "metric_path": str(metric_path),
            }
        )

    scene_rows.sort(key=lambda row: str(row["scene"]))
    summary_csv = root / "campaign_summary.csv"
    fieldnames = [
        "scene",
        "episodes",
        "unique_episodes",
        "successes",
        "sr",
        "spl",
        "collision_rate",
        "collision_free_sr",
        "mean_linear_accel_rms",
        "mean_trajectory_length",
        "mean_elapsed_time",
        "complete",
        "metric_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scene_rows)

    completed_scenes = sum(int(row["complete"]) for row in scene_rows)
    expected_total = args.expected_scenes * args.expected_episodes
    report_lines = [
        f"Campaign: {root}",
        f"Scenes found: {len(scene_rows)}/{args.expected_scenes}",
        f"Scenes complete: {completed_scenes}/{args.expected_scenes}",
        f"Episodes: {len(all_rows)}/{expected_total}",
        f"Successes: {sum(row['success'] >= 0.5 for row in all_rows)}",
        f"SR: {mean([row['success'] for row in all_rows]):.6f}",
        f"SPL: {mean([row['spl'] for row in all_rows]):.6f}",
        f"Collision rate: {mean([row['collision'] for row in all_rows]):.6f}",
        "Collision-free SR: "
        f"{mean([float(row['success'] >= 0.5 and row['collision'] < 0.5) for row in all_rows]):.6f}",
        "Mean linear acceleration RMS (m/s^2, lower is smoother): "
        f"{finite_mean([row['linear_accel_rms'] for row in all_rows]):.6f}",
        f"Mean trajectory length: "
        f"{mean([row['trajectory_length'] for row in all_rows]):.6f}",
        f"Mean elapsed time: {mean([row['elapsed_time'] for row in all_rows]):.6f}",
        f"Complete: {int(completed_scenes == args.expected_scenes and len(all_rows) == expected_total)}",
        f"Per-scene CSV: {summary_csv}",
    ]
    report = "\n".join(report_lines) + "\n"
    (root / "campaign_summary.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
