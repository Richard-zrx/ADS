from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import datasets
import pandas as pd

# Default math-oriented prompt template. Each question is wrapped into a single
# `user` turn that asks for step-by-step reasoning and a \boxed{} final answer.
# Adjust this template if your task is not boxed-answer math.
PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "{question}\n\n"
    "<think>"
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an arbitrary math dataset into the unified VeRL/GRPO format. "
        "Bring your own data: pass a Hugging Face Hub id or a local parquet/json file."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Source dataset: a Hugging Face Hub id (e.g. 'org/name') or a local "
        "file path (.parquet / .json / .jsonl).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to load when --dataset is a Hugging Face Hub id (default: train).",
    )
    parser.add_argument("--output-root", default=str(Path.cwd() / "dataset"))
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output subdirectory name under --output-root. Defaults to a sanitized "
        "form of --dataset (with a frac/seed suffix when --fraction < 1.0).",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Keep only this fraction of rows as a deterministic random subset "
        "(0 < f <= 1). Default 1.0 keeps every row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to draw the subset when --fraction < 1.0.",
    )
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    """Turn a Hugging Face id or file path into a filesystem-friendly name."""
    base = text.rstrip("/").split("/")[-1]
    base = re.sub(r"\.(parquet|json|jsonl)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return base or "dataset"


def resolve_output_name(dataset: str, fraction: float, seed: int, explicit: str | None) -> str:
    if explicit:
        return explicit
    base = sanitize_name(dataset)
    if fraction >= 1.0:
        return base
    pct = f"{fraction:g}".replace("0.", "")
    return f"{base}_frac{pct}_seed{seed}"


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text = part
            elif isinstance(part, dict):
                text = to_text(part.get("text") if part.get("text") is not None else part.get("content"))
            else:
                text = to_text(part)
            text = text.strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        return to_text(content.get("text") if content.get("text") is not None else content.get("content"))
    return to_text(content)


def normalize_prompt_messages(prompt_value: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(prompt_value, list):
        iterable = prompt_value
    elif prompt_value is None:
        iterable = []
    else:
        iterable = [prompt_value]

    for item in iterable:
        if isinstance(item, dict):
            role = to_text(item.get("role")).strip().lower()
            content = flatten_content(item.get("content")).strip()
        else:
            role = ""
            content = flatten_content(item).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def looks_like_instruction(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "reason step by step",
        "final answer",
        "\\boxed{}",
        "boxed{}",
        "assistant",
        "math problem",
        "question:",
    )
    return any(marker in lowered for marker in markers)


def strip_instruction_prefix(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").strip()
    text = re.sub(r"\s*<think>\s*$", "", text, flags=re.IGNORECASE).strip()

    if "\n\n" in text:
        first, rest = text.split("\n\n", 1)
        if looks_like_instruction(first) and rest.strip():
            text = rest.strip()

    for tag in ("Question:", "Problem:", "Q:"):
        idx = text.lower().rfind(tag.lower())
        if idx > 0 and looks_like_instruction(text[:idx]):
            candidate = text[idx + len(tag) :].strip()
            if candidate:
                text = candidate
                break

    for tag in ("Question:", "Problem:", "Q:"):
        if text.lower().startswith(tag.lower()):
            stripped = text[len(tag) :].strip()
            if stripped:
                text = stripped
                break

    text = re.sub(r"\s*<think>\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def extract_question_and_raw(example: dict[str, Any], idx: int) -> tuple[str, str]:
    messages = normalize_prompt_messages(example.get("prompt"))
    user_like = [
        message["content"]
        for message in messages
        if message["role"] in {"user", "human"} or "user" in message["role"] or "human" in message["role"]
    ]

    if user_like:
        raw = user_like[-1].strip()
    elif messages:
        raw = messages[-1]["content"].strip()
    else:
        fallback_fields = ("question", "problem", "input")
        raw = ""
        for field in fallback_fields:
            candidate = to_text(example.get(field)).strip()
            if candidate:
                raw = candidate
                break
        if not raw:
            raise ValueError(f"Unable to extract question text at row {idx}")

    cleaned = strip_instruction_prefix(raw)
    if not cleaned:
        cleaned = raw.strip()
    if not cleaned:
        raise ValueError(f"Empty cleaned question at row {idx}")
    return cleaned, raw


def extract_ground_truth(example: dict[str, Any], idx: int) -> str:
    reward_model = example.get("reward_model")
    ground_truth = ""
    if isinstance(reward_model, dict):
        ground_truth = to_text(reward_model.get("ground_truth")).strip()
    if not ground_truth:
        # Fall back to common answer fields when reward_model is absent.
        for field in ("ground_truth", "answer", "final_answer", "solution"):
            candidate = to_text(example.get(field)).strip()
            if candidate:
                ground_truth = candidate
                break
    if not ground_truth:
        raise ValueError(f"Missing ground-truth answer at row {idx}")
    return ground_truth


def extract_target_text(example: dict[str, Any]) -> str:
    target = example.get("target")
    if target is None:
        return ""
    if isinstance(target, list):
        parts: list[str] = []
        for item in target:
            if isinstance(item, dict):
                part = flatten_content(item.get("content"))
            else:
                part = flatten_content(item)
            part = part.strip()
            if part:
                parts.append(part)
        return "\n".join(parts).strip()
    if isinstance(target, dict):
        return flatten_content(target.get("content")).strip()
    return to_text(target).strip()


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


def load_source_dataset(dataset: str, split: str, cache_dir: str | None) -> tuple[datasets.Dataset, str]:
    """Load from a local parquet/json/jsonl file or from the Hugging Face Hub.

    Returns the dataset and the split label recorded in extra_info.
    """
    local_path = Path(dataset)
    if local_path.exists():
        suffix = local_path.suffix.lower().lstrip(".")
        builder = {"parquet": "parquet", "json": "json", "jsonl": "json"}.get(suffix)
        if builder is None:
            raise ValueError(f"Unsupported local file type '.{suffix}'. Use .parquet / .json / .jsonl.")
        ds = datasets.load_dataset(builder, data_files=str(local_path), split="train", cache_dir=cache_dir)
        return ds, "local"
    ds = datasets.load_dataset(dataset, split=split, cache_dir=cache_dir)
    return ds, split


def main() -> None:
    args = parse_args()
    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    output_root = Path(args.output_root)
    output_name = resolve_output_name(args.dataset, args.fraction, args.seed, args.output_name)
    default_source = sanitize_name(args.dataset)
    output_dir = output_root / output_name
    parquet_path = output_dir / "train.parquet"
    example_path = output_dir / "example.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_ds, split_label = load_source_dataset(args.dataset, args.split, args.cache_dir)
    total_rows = len(source_ds)

    # Deterministic random subset (sorted to preserve original ordering, so row-index
    # sample_ids stay stable and reproducible across pipeline phases).
    selected_indices = list(range(total_rows))
    if args.fraction < 1.0:
        import random

        keep = max(1, round(args.fraction * total_rows))
        selected_indices = sorted(random.Random(args.seed).sample(range(total_rows), keep))
    expected_rows = len(selected_indices)

    records: list[dict[str, Any]] = []
    for idx in selected_indices:
        example = source_ds[idx]
        question, raw_question = extract_question_and_raw(example, idx)
        ground_truth = extract_ground_truth(example, idx)
        target_text = extract_target_text(example)

        source_id = (
            to_text(example.get("sample_id")).strip()
            or to_text(example.get("id")).strip()
            or to_text(example.get("uid")).strip()
            or str(idx)
        )

        data_source = to_text(example.get("data_source")).strip() or default_source
        raw_answer = ground_truth if ground_truth else target_text

        record = {
            "data_source": data_source,
            "prompt": build_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            # Carry the chain-of-thought so the clustering/difficulty stage can use
            # input_text_mode=question_plus_cot. Empty when the source has no CoT.
            "target": target_text,
            "extra_info": {
                "source_dataset": args.dataset,
                "source_split": split_label,
                "source_id": source_id,
                "subject": None,
                "level": None,
                "year": None,
                "url": None,
                "raw_question": raw_question,
                "raw_answer": raw_answer,
                "has_solution": bool(target_text),
            },
        }
        validate_record(record, idx)
        records.append(record)

    if len(records) != expected_rows:
        raise ValueError(f"Row count mismatch before write: expected={expected_rows} output={len(records)}")

    datasets.Dataset.from_list(records).to_parquet(str(parquet_path))
    output_rows = int(pd.read_parquet(parquet_path).shape[0])
    if output_rows != expected_rows:
        raise ValueError(f"Row count mismatch after write: expected={expected_rows} parquet={output_rows}")

    with example_path.open("w", encoding="utf-8") as f:
        json.dump(records[0], f, ensure_ascii=False, indent=2)

    print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print(f"[{output_name}] source_rows={total_rows} fraction={args.fraction} output_rows={output_rows}")
    print(f"[{output_name}] parquet={parquet_path}")
    print(f"[{output_name}] example={example_path}")


if __name__ == "__main__":
    main()
