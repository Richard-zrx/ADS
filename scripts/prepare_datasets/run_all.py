#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the evaluation sets (AIME 2024, AIME 2025, MATH-500) and build summary.json. "
        "For the training set, bring your own data with prepare_custom_dataset.py."
    )
    parser.add_argument("--output-root", default=str(Path.cwd() / "dataset"))
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    scripts_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)

    output_dirs = [
        output_root,
        output_root / "math500",
        output_root / "aime25",
        output_root / "aime2024",
    ]
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "name": "math500",
            "script": "prepare_math500.py",
            "split": "test",
            "parquet": output_root / "math500" / "test.parquet",
            "example": output_root / "math500" / "example.json",
        },
        {
            "name": "aime25",
            "script": "prepare_aime25.py",
            "split": "test",
            "parquet": output_root / "aime25" / "test.parquet",
            "example": output_root / "aime25" / "example.json",
        },
        {
            "name": "aime2024",
            "script": "prepare_aime2024.py",
            "split": "train->test",
            "parquet": output_root / "aime2024" / "test.parquet",
            "example": output_root / "aime2024" / "example.json",
        },
    ]

    summary: dict[str, dict[str, object]] = {}
    for job in jobs:
        cmd = [
            sys.executable,
            str(scripts_dir / job["script"]),
            "--output-root",
            str(output_root),
        ]
        if args.cache_dir:
            cmd.extend(["--cache-dir", str(args.cache_dir)])
        cmd.extend(job.get("extra_args", []))

        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=repo_root)

        parquet_path = Path(job["parquet"])
        example_path = Path(job["example"])
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing output parquet: {parquet_path}")
        if not example_path.exists():
            raise FileNotFoundError(f"Missing output example: {example_path}")

        rows = int(pd.read_parquet(parquet_path).shape[0])
        try:
            output_ref = str(parquet_path.relative_to(repo_root))
        except ValueError:
            output_ref = str(parquet_path)

        summary[str(job["name"])] = {
            "rows": rows,
            "split": job["split"],
            "output": output_ref,
        }

        print(f"[{job['name']}] output={parquet_path}", flush=True)
        print(f"[{job['name']}] rows={rows}", flush=True)
        print(f"[{job['name']}] example={example_path}", flush=True)

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
