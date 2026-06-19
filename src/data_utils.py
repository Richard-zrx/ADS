from __future__ import annotations

import json
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset


def load_raw_dataset(input_path: str, input_format: str, cache_dir: str | None = None) -> Dataset:
    fmt = input_format.strip().lower()
    if fmt == "parquet":
        return load_dataset("parquet", data_files={"train": input_path}, split="train", cache_dir=cache_dir)
    if fmt in {"jsonl", "json"}:
        return load_dataset("json", data_files={"train": input_path}, split="train", cache_dir=cache_dir)
    raise ValueError(f"Unsupported input_format: {input_format}")


def _safe_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _to_message_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, dict):
        if "content" in value:
            return [value]
        return [{"role": "user", "content": _safe_json_or_str(value)}]

    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if isinstance(value, list):
        messages: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                messages.append(item)
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
            else:
                messages.append({"role": "user", "content": _safe_json_or_str(item)})
        return messages

    return [{"role": "user", "content": str(value)}]


def _join_contents(messages: list[dict[str, Any]], roles: set[str] | None = None) -> str:
    chunks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        if roles is not None and role not in roles:
            continue
        content = msg.get("content", "")
        text = str(content).strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _build_question_text(example: dict[str, Any], text_field: str, prompt_text_mode: str) -> str:
    raw_prompt = example.get(text_field)
    if raw_prompt is None and text_field != "prompt":
        raw_prompt = example.get("prompt")

    messages = _to_message_list(raw_prompt)
    mode = prompt_text_mode.strip().lower()

    if mode == "user_only":
        user_text = _join_contents(messages, roles={"user"})
        if user_text:
            return user_text
        fallback = _join_contents(messages, roles=None)
        return fallback if fallback else ""

    if mode in {"all", "system_user"}:
        merged = _join_contents(messages, roles=None)
        return merged if merged else ""

    raise ValueError(f"Unsupported prompt_text_mode: {prompt_text_mode}")


def _extract_ground_truth(example: dict[str, Any]) -> str:
    reward_model = example.get("reward_model")
    if isinstance(reward_model, np.ndarray):
        reward_model = reward_model.tolist()

    ground_truth = None
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")

    if ground_truth is None:
        raise ValueError("Missing reward_model.ground_truth for question_plus_ground_truth input mode")

    text = str(ground_truth).strip()
    if not text:
        raise ValueError("Empty reward_model.ground_truth for question_plus_ground_truth input mode")
    return text


def _extract_cot_answer(example: dict[str, Any]) -> str:
    """Extract full chain-of-thought from the target field."""
    target = example.get("target")
    if isinstance(target, np.ndarray):
        target = target.tolist()
    if isinstance(target, list) and len(target) > 0:
        item = target[0]
        if isinstance(item, dict):
            content = item.get("content", "")
            text = str(content).strip()
            if text:
                return text
    if isinstance(target, str):
        text = target.strip()
        if text:
            return text
    raise ValueError("Missing or empty target field for question_plus_cot input mode")


def build_input_text(
    example: dict[str, Any],
    text_field: str,
    prompt_text_mode: str,
    input_text_mode: str = "question_only",
) -> str:
    question_text = _build_question_text(example, text_field=text_field, prompt_text_mode=prompt_text_mode)
    mode = input_text_mode.strip().lower()

    if mode == "question_only":
        return question_text

    if mode == "question_plus_ground_truth":
        ground_truth = _extract_ground_truth(example)
        return f"Question:\n{question_text}\n\nAnswer:\n{ground_truth}"

    if mode == "question_plus_cot":
        cot_answer = _extract_cot_answer(example)
        return f"Question:\n{question_text}\n\nAnswer:\n{cot_answer}"

    raise ValueError(f"Unsupported input_text_mode: {input_text_mode}")


def _normalize_sample_id(value: Any, default_idx: int) -> Any:
    if value is None:
        return int(default_idx)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = str(value).strip()
    if text == "":
        return int(default_idx)
    try:
        return int(text)
    except ValueError:
        return text


def ensure_sample_id(example: dict[str, Any], idx: int, id_field: str) -> Any:
    direct = example.get(id_field)
    if direct is not None:
        return _normalize_sample_id(direct, idx)

    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        normalized_index = _normalize_sample_id(extra_info.get("index"), idx)
        # Some datasets use sentinel index=-1 for all rows.
        # Treat negative numeric index as invalid and fall back to row idx.
        if isinstance(normalized_index, (int, np.integer)):
            if int(normalized_index) >= 0:
                return int(normalized_index)
        else:
            return normalized_index

    return int(idx)


def _map_example(
    example: dict[str, Any],
    idx: int,
    text_field: str,
    id_field: str,
    prompt_text_mode: str,
    input_text_mode: str,
) -> dict[str, Any]:
    sample_id = ensure_sample_id(example, idx, id_field=id_field)
    input_text = build_input_text(
        example,
        text_field=text_field,
        prompt_text_mode=prompt_text_mode,
        input_text_mode=input_text_mode,
    )
    return {
        "sample_id": sample_id,
        "input_text": input_text,
    }


def _safe_dataset_map(
    dataset: Dataset,
    map_fn,
    fn_kwargs: dict[str, Any],
    num_proc: int,
) -> Dataset:
    map_num_proc = num_proc if num_proc and num_proc > 1 else None
    map_kwargs = {
        "with_indices": True,
        "fn_kwargs": fn_kwargs,
        "num_proc": map_num_proc,
    }
    try:
        return dataset.map(map_fn, **map_kwargs)
    except (PermissionError, OSError):
        if map_num_proc is None:
            raise
        map_kwargs["num_proc"] = None
        return dataset.map(map_fn, **map_kwargs)


def prepare_embedding_dataset(
    dataset: Dataset,
    text_field: str,
    id_field: str,
    prompt_text_mode: str,
    input_text_mode: str,
    num_proc: int,
) -> Dataset:
    mapped = _safe_dataset_map(
        dataset=dataset,
        map_fn=_map_example,
        fn_kwargs={
            "text_field": text_field,
            "id_field": id_field,
            "prompt_text_mode": prompt_text_mode,
            "input_text_mode": input_text_mode,
        },
        num_proc=num_proc,
    )

    keep_cols = {"sample_id", "input_text"}
    remove_cols = [col for col in mapped.column_names if col not in keep_cols]
    if remove_cols:
        mapped = mapped.remove_columns(remove_cols)
    return mapped


def load_and_prepare_dataset(config: dict[str, Any]) -> Dataset:
    cache_dir = config.get("datasets_cache_dir")
    if cache_dir is not None:
        cache_dir = str(cache_dir)

    raw_dataset = load_raw_dataset(
        input_path=str(config["input_path"]),
        input_format=str(config.get("input_format", "parquet")),
        cache_dir=cache_dir,
    )
    prepared = prepare_embedding_dataset(
        dataset=raw_dataset,
        text_field=str(config.get("text_field", "prompt")),
        id_field=str(config.get("id_field", "sample_id")),
        prompt_text_mode=str(config.get("prompt_text_mode", "user_only")),
        input_text_mode=str(config.get("input_text_mode", "question_only")),
        num_proc=int(config.get("num_proc", 1)),
    )
    return prepared


def _map_record_for_lookup(
    example: dict[str, Any],
    idx: int,
    text_field: str,
    id_field: str,
    prompt_text_mode: str,
    input_text_mode: str,
) -> dict[str, Any]:
    sample_id = ensure_sample_id(example, idx, id_field=id_field)
    input_text = build_input_text(
        example,
        text_field=text_field,
        prompt_text_mode=prompt_text_mode,
        input_text_mode=input_text_mode,
    )
    target = example.get("target")
    extra_info = example.get("extra_info")
    source_split = None
    if isinstance(extra_info, dict):
        source_split = extra_info.get("split")

    return {
        "sample_id": sample_id,
        "input_text": input_text,
        "target": target,
        "extra_info": extra_info,
        "source_split": source_split,
    }


def build_id_to_text_map(
    raw_data_path: str,
    raw_data_format: str,
    text_field: str,
    id_field: str,
    prompt_text_mode: str,
    input_text_mode: str,
    num_proc: int,
    cache_dir: str | None = None,
) -> dict[Any, dict[str, Any]]:
    dataset = load_raw_dataset(raw_data_path, raw_data_format, cache_dir=cache_dir)
    mapped = _safe_dataset_map(
        dataset=dataset,
        map_fn=_map_record_for_lookup,
        fn_kwargs={
            "text_field": text_field,
            "id_field": id_field,
            "prompt_text_mode": prompt_text_mode,
            "input_text_mode": input_text_mode,
        },
        num_proc=num_proc,
    )

    id_to_record: dict[Any, dict[str, Any]] = {}
    for row in mapped:
        sample_id = row["sample_id"]
        if sample_id in id_to_record:
            raise ValueError(f"Duplicate sample_id found in raw dataset mapping: {sample_id}")
        id_to_record[sample_id] = {
            "input_text": row.get("input_text", ""),
            "target": row.get("target"),
            "extra_info": row.get("extra_info"),
            "source_split": row.get("source_split"),
        }
    return id_to_record
