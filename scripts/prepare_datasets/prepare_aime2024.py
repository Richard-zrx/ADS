from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import datasets
import pandas as pd

PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "{question}\n\n"
    "<think>"
)
DATASET_NAME = "HuggingFaceH4/aime_2024"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HuggingFaceH4/aime_2024 into unified VeRL/GRPO format.")
    parser.add_argument("--output-root", default=str(Path.cwd() / "dataset"))
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def to_optional_text(value: Any) -> str | None:
    text = to_text(value).strip()
    return text if text else None


def build_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question.strip())}]


def validate_record(record: dict[str, Any], idx: int) -> None:
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"Row {idx}: prompt must be a list")
    if len(prompt) != 1:
        raise ValueError(f"Row {idx}: prompt length must be 1")
    role = to_text(prompt[0].get("role")).strip()
    if role != "user":
        raise ValueError(f"Row {idx}: prompt role must be user")

    content = to_text(prompt[0].get("content"))
    if "Please reason step by step" not in content:
        raise ValueError(f"Row {idx}: prompt missing instruction phrase")
    if "\\boxed{}" not in content:
        raise ValueError(f"Row {idx}: prompt missing \\\\boxed{{}}")
    if not content.rstrip().endswith("<think>"):
        raise ValueError(f"Row {idx}: prompt must end with <think>")

    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError(f"Row {idx}: reward_model must be a dict")
    ground_truth = to_text(reward_model.get("ground_truth")).strip()
    if not ground_truth:
        raise ValueError(f"Row {idx}: reward_model.ground_truth must be non-empty")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_dir = output_root / "aime2024"
    parquet_path = output_dir / "test.parquet"
    example_path = output_dir / "example.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_ds = datasets.load_dataset(DATASET_NAME, split="train", cache_dir=args.cache_dir)
    input_rows = len(source_ds)
    records: list[dict[str, Any]] = []

    for idx in range(input_rows):
        example = source_ds[idx]
        question = to_text(example.get("problem")).strip()
        if not question:
            raise ValueError(f"Row {idx}: empty problem")

        ground_truth = to_text(example.get("answer")).strip()
        if not ground_truth:
            raise ValueError(f"Row {idx}: empty answer")

        source_id = to_text(example.get("id")).strip() or str(idx)
        year = to_optional_text(example.get("year"))
        url = to_optional_text(example.get("url"))
        record = {
            "data_source": "aime2024",
            "prompt": build_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": {
                "source_dataset": DATASET_NAME,
                "source_split": "train",
                "source_id": source_id,
                "subject": None,
                "level": None,
                "year": year,
                "url": url,
                "raw_question": question,
                "raw_answer": ground_truth,
                "has_solution": True,
            },
        }
        validate_record(record, idx)
        records.append(record)

    if len(records) != input_rows:
        raise ValueError(f"Row count mismatch before write: input={input_rows} output={len(records)}")

    datasets.Dataset.from_list(records).to_parquet(str(parquet_path))
    output_rows = int(pd.read_parquet(parquet_path).shape[0])
    if output_rows != input_rows:
        raise ValueError(f"Row count mismatch after write: input={input_rows} parquet={output_rows}")

    with example_path.open("w", encoding="utf-8") as f:
        json.dump(records[0], f, ensure_ascii=False, indent=2)

    print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print(f"[aime2024] input_rows={input_rows} output_rows={output_rows}")
    print(f"[aime2024] parquet={parquet_path}")
    print(f"[aime2024] example={example_path}")


if __name__ == "__main__":
    main()
