# Dataset Preparation for VeRL / GRPO

These scripts convert datasets into one unified schema for VeRL/GRPO training and evaluation. `prepare_custom_dataset.py` formats any math dataset (Hugging Face Hub id or local file) into the schema below. The three evaluation benchmarks (AIME 2024, AIME 2025, MATH-500) have dedicated scripts.

## Output Directories

All generated files are written under `dataset/`:

- `dataset/<your_dataset>/` — your formatted training set
- `dataset/math500/`
- `dataset/aime25/`
- `dataset/aime2024/`
- `dataset/summary.json` (eval sets)

## Training set

`prepare_custom_dataset.py` reads a Hugging Face Hub dataset **or** a local `.parquet` / `.json` / `.jsonl` file and writes `dataset/<name>/train.parquet` in the unified schema:

```bash
# from a Hugging Face Hub dataset
python scripts/prepare_datasets/prepare_custom_dataset.py --dataset <org/name> --split train

# from a local file
python scripts/prepare_datasets/prepare_custom_dataset.py --dataset path/to/train.parquet
```

## Evaluation sets

```bash
python scripts/prepare_datasets/prepare_math500.py
python scripts/prepare_datasets/prepare_aime25.py
python scripts/prepare_datasets/prepare_aime2024.py
```

### Run all eval sets at once

```bash
python scripts/prepare_datasets/run_all.py
```

`run_all.py` runs the three evaluation scripts sequentially and writes `dataset/summary.json`.

## Unified Output Schema

All output parquet rows use:

```json
{
  "data_source": "string",
  "prompt": [
    {
      "role": "user",
      "content": "string"
    }
  ],
  "ability": "math",
  "reward_model": {
    "ground_truth": "string",
    "style": "rule"
  },
  "target": "string",
  "extra_info": {
    "source_dataset": "string",
    "source_split": "string",
    "source_id": "string",
    "subject": "string|null",
    "level": "int|null",
    "year": "string|null",
    "url": "string|null",
    "raw_question": "string",
    "raw_answer": "string",
    "has_solution": "bool"
  }
}
```

## Prompt example

```text
Please reason step by step, and put your final answer within \boxed{}.

{question}

<think>
```
