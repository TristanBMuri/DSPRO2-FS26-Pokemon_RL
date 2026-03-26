#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import mlflow
from dotenv import find_dotenv, load_dotenv

from src.validation.trainer_validation import run_validation


def write_artifacts(temp_dir: Path, result: dict) -> None:
    rows = result["rows"]
    summary = result["summary"]
    dataset_meta = result["dataset_meta"]

    summary_path = temp_dir / "validation_summary.json"
    rows_path = temp_dir / "validation_rows.csv"

    summary_payload = {
        "dataset_meta": dataset_meta,
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with rows_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows_path.write_text("", encoding="utf-8")


def main() -> int:
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description="Run validation against BDSP trainer dataset and log to MLflow.")
    parser.add_argument("--checkpoint", default="checkpoints/final", help="Path to RLlib checkpoint directory")
    parser.add_argument("--dataset", default="data/bdsp_trainers.json", help="Path to trainer dataset JSON")
    parser.add_argument("--preset", default="quick", help="Model/config preset used for restore")
    parser.add_argument("--host", default="127.0.0.1", help="Showdown host")
    parser.add_argument("--port", type=int, default=8000, help="Showdown port")
    parser.add_argument("--limit", type=int, default=10, help="How many trainers to validate (smoke-test default: 10)")
    parser.add_argument("--episodes-per-trainer", type=int, default=1, help="Episodes per trainer")
    parser.add_argument("--opponent-policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--experiment-name", default="Pokemon_RL_Validation")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.set_tag("run_type", "validation")
        mlflow.set_tag("dataset", str(args.dataset))
        mlflow.set_tag("checkpoint", str(args.checkpoint))

        mlflow.log_params({
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "preset": args.preset,
            "showdown_host": args.host,
            "showdown_port": args.port,
            "limit": args.limit,
            "episodes_per_trainer": args.episodes_per_trainer,
            "opponent_policy": args.opponent_policy,
        })

        result = run_validation(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset,
            preset=args.preset,
            host=args.host,
            port=args.port,
            limit=args.limit,
            episodes_per_trainer=args.episodes_per_trainer,
            opponent_policy=args.opponent_policy,
            strict_dataset=False,
        )

        mlflow.log_metrics(result["summary"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_artifacts(tmp_path, result)
            mlflow.log_artifacts(str(tmp_path), artifact_path="validation")

        print("=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        for k, v in result["summary"].items():
            print(f"{k}: {v}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
