#!/usr/bin/env python3
"""
Run AquaCrisis LLM classification experiments with resumable JSONL checkpoints.

This script is designed for Mahti/Slurm runs, but it also works locally.
Each experiment writes JSONL, JSON, and CSV prediction files. If a job stops,
rerun the same command with --resume and already completed post_ids are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TASK_A_LABELS = {
    "A1": "Water quality / safety / public health",
    "A2": "Service disruption / repair / infrastructure",
    "A3": "Water conservation / drought / demand management",
    "A4": "Flood / stormwater / wastewater / sewer",
    "A5": "Environmental sustainability / waste / recycling / biodiversity",
    "A6": "Public education / heritage / community engagement",
    "A7": "Routine / institutional / customer service / other",
}

TASK_B_LABELS = {
    "B1": "Alert / warning",
    "B2": "Instruction / advice to public",
    "B3": "Operational update / resolution",
    "B4": "Reassurance / safety information",
    "B5": "Education / awareness",
    "B6": "Institutional promotion / community news",
    "B7": "Other / unclear",
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    provider: str
    model: str
    shot: str
    text_source: str
    text_col: str


EXPERIMENTS = [
    Experiment("qwen35_zero_original", "ollama", "qwen3.5:9b", "zero", "original", "native_text"),
    Experiment("qwen35_zero_translated", "ollama", "qwen3.5:9b", "zero", "translated", "translated_text"),
    Experiment("qwen35_five_original", "ollama", "qwen3.5:9b", "five", "original", "native_text"),
    Experiment("qwen35_five_translated", "ollama", "qwen3.5:9b", "five", "translated", "translated_text"),
    Experiment("gemma4_zero_original", "ollama", "gemma4:e4b", "zero", "original", "native_text"),
    Experiment("gemma4_zero_translated", "ollama", "gemma4:e4b", "zero", "translated", "translated_text"),
    Experiment("gemma4_five_original", "ollama", "gemma4:e4b", "five", "original", "native_text"),
    Experiment("gemma4_five_translated", "ollama", "gemma4:e4b", "five", "translated", "translated_text"),
    Experiment("gpt41mini_zero_original", "openai", "gpt-4.1-mini", "zero", "original", "native_text"),
    Experiment("gpt41mini_zero_translated", "openai", "gpt-4.1-mini", "zero", "translated", "translated_text"),
    Experiment("gpt41mini_five_original", "openai", "gpt-4.1-mini", "five", "original", "native_text"),
    Experiment("gpt41mini_five_translated", "openai", "gpt-4.1-mini", "five", "translated", "translated_text"),
    Experiment("gpt4omini_zero_original", "openai", "gpt-4o-mini", "zero", "original", "native_text"),
    Experiment("gpt4omini_zero_translated", "openai", "gpt-4o-mini", "zero", "translated", "translated_text"),
    Experiment("gpt4omini_five_original", "openai", "gpt-4o-mini", "five", "original", "native_text"),
    Experiment("gpt4omini_five_translated", "openai", "gpt-4o-mini", "five", "translated", "translated_text"),
]

EXPERIMENT_BY_ID = {exp.experiment_id: exp for exp in EXPERIMENTS}


FIVE_SHOT_EXAMPLES = [
    {
        "text": (
            "Due to possible bacterial contamination, residents in the affected area "
            "must boil tap water before drinking or cooking until further notice."
        ),
        "output": {
            "task_a": {"label_id": "A1", "label_name": TASK_A_LABELS["A1"]},
            "task_b": {"label_id": "B2", "label_name": TASK_B_LABELS["B2"]},
            "confidence": 0.98,
        },
    },
    {
        "text": (
            "Repair crews are working on a burst water pipe. Water pressure is low, "
            "and service is expected to return this evening."
        ),
        "output": {
            "task_a": {"label_id": "A2", "label_name": TASK_A_LABELS["A2"]},
            "task_b": {"label_id": "B3", "label_name": TASK_B_LABELS["B3"]},
            "confidence": 0.96,
        },
    },
    {
        "text": (
            "During the dry and hot period, please save water by avoiding garden "
            "watering during the day and using water wisely."
        ),
        "output": {
            "task_a": {"label_id": "A3", "label_name": TASK_A_LABELS["A3"]},
            "task_b": {"label_id": "B2", "label_name": TASK_B_LABELS["B2"]},
            "confidence": 0.95,
        },
    },
    {
        "text": (
            "Heavy rain may overload drains and cause sewer overflow in low-lying "
            "areas. Avoid flooded streets and follow local warnings."
        ),
        "output": {
            "task_a": {"label_id": "A4", "label_name": TASK_A_LABELS["A4"]},
            "task_b": {"label_id": "B1", "label_name": TASK_B_LABELS["B1"]},
            "confidence": 0.95,
        },
    },
    {
        "text": (
            "Learn why correct waste sorting and recycling protect rivers, wildlife, "
            "and local biodiversity."
        ),
        "output": {
            "task_a": {"label_id": "A5", "label_name": TASK_A_LABELS["A5"]},
            "task_b": {"label_id": "B5", "label_name": TASK_B_LABELS["B5"]},
            "confidence": 0.94,
        },
    },
]


SYSTEM_PROMPT = """You are a careful text classification assistant. Your task is to assign exactly one label for Task A and exactly one label for Task B.

Follow these rules:
- Use only the label IDs and names listed below.
- Choose the single best label for each task.
- Do not output explanations.
- Return valid JSON only.
- If the post is short or ambiguous, choose the most plausible label using the definitions below.

Task A labels:
A1 Water quality / safety / public health
Posts about water safety, drinking-water quality, contamination, testing, boil-water notices, bacteria, chemical risks, turbidity, chlorine, PFAS, or health risk.

A2 Service disruption / repair / infrastructure
Posts about outages, interruptions, repairs, leaks, bursts, maintenance, restoration, crews, pipelines, treatment plants, or infrastructure problems.

A3 Water conservation / drought / demand management
Posts about saving water, drought, restrictions, efficient use, demand reduction, conservation campaigns, or scarcity.

A4 Flood / stormwater / wastewater / sewer
Posts about flooding, heavy rain response, drains, sewer overflow, stormwater, wastewater incidents, or sewage.

A5 Environmental sustainability / waste / recycling / biodiversity
Posts about environmental protection, emissions, habitat, recycling, biodiversity, wetlands, river ecology, sustainability programs, or nature stewardship.

A6 Public education / heritage / community engagement
Posts about educational activities, awareness events, tours, school visits, museum/heritage content, public exhibitions, or community engagement not mainly about a current operational issue.

A7 Routine / institutional / customer service / other
Posts about office hours, billing, job vacancies, holiday notices, generic greetings, routine institutional communication, or items that do not fit A1-A6.

Task B labels:
B1 Alert / warning
Urgent alert or warning about a risk, disruption, incident, hazard, or emergency.

B2 Instruction / advice to public
Tells people what to do, for example boil water, avoid an area, conserve water, prepare, report leaks, or follow safety advice.

B3 Operational update / resolution
Status update about work underway, service restoration, incident handling, repair progress, or completion.

B4 Reassurance / safety information
Explains that water is safe, risks are low, monitoring is ongoing, or provides calming safety clarification.

B5 Education / awareness
Explains information for public understanding without mainly promoting the institution.

B6 Institutional promotion / community news
Promotes events, campaigns, achievements, awards, partnerships, or general community news.

B7 Other / unclear
Use when the communication function is too vague, ambiguous, context-dependent, or does not fit B1-B6.

Return this JSON schema exactly:
{
  "task_a": {"label_id": "...", "label_name": "..."},
  "task_b": {"label_id": "...", "label_name": "..."},
  "confidence": 0.00
}
"""


BATCH_OUTPUT_INSTRUCTIONS = """Classify each post independently.

Return valid JSON only, with this exact top-level schema:
{
  "predictions": [
    {
      "post_id": "...",
      "task_a": {"label_id": "...", "label_name": "..."},
      "task_b": {"label_id": "...", "label_name": "..."},
      "confidence": 0.00
    }
  ]
}

Rules for batch output:
- Return exactly one prediction object for each input post.
- Use the exact post_id from the input.
- Do not add extra post_ids.
- Do not skip any input posts.
- Do not output explanations.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_messages(text: str, shot: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if shot == "five":
        for i, example in enumerate(FIVE_SHOT_EXAMPLES, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": f"Example {i}\nText: {example['text']}\nOutput:",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example["output"], ensure_ascii=False, separators=(",", ":")),
                }
            )

        messages.append({"role": "user", "content": f"Now classify this new post:\n{text}"})
    else:
        messages.append({"role": "user", "content": f"Classify this post:\n{text}"})
    return messages


def make_batch_messages(posts: list[dict[str, str]], shot: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_INSTRUCTIONS}"}]
    if shot == "five":
        for i, example in enumerate(FIVE_SHOT_EXAMPLES, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": f"Example {i}\nText: {example['text']}\nOutput:",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example["output"], ensure_ascii=False, separators=(",", ":")),
                }
            )

    input_posts = [
        {
            "post_id": str(post["post_id"]),
            "text": str(post["text"]),
        }
        for post in posts
    ]
    messages.append(
        {
            "role": "user",
            "content": (
                "Classify these posts and return the batch JSON object described above.\n"
                f"Posts:\n{json.dumps(input_posts, ensure_ascii=False, separators=(',', ':'))}"
            ),
        }
    )
    return messages


def extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def validate_batch_predictions(
    parsed: dict[str, Any],
    expected_post_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    predictions = parsed.get("predictions")
    if not isinstance(predictions, list):
        return {}, ["Missing or invalid top-level predictions list"]

    expected = [str(post_id) for post_id in expected_post_ids]
    expected_set = set(expected)
    by_post_id: dict[str, dict[str, Any]] = {}

    for item in predictions:
        if not isinstance(item, dict):
            errors.append("Prediction item is not an object")
            continue
        post_id = str(item.get("post_id", "")).strip()
        if not post_id:
            errors.append("Prediction item missing post_id")
            continue
        if post_id in by_post_id:
            errors.append(f"Duplicate prediction for post_id: {post_id}")
            continue
        normalized, validation_errors = validate_prediction(item)
        if validation_errors:
            errors.append(f"{post_id}: {'; '.join(validation_errors)}")
        by_post_id[post_id] = normalized

    missing = [post_id for post_id in expected if post_id not in by_post_id]
    extra = [post_id for post_id in by_post_id if post_id not in expected_set]
    if missing:
        errors.append(f"Missing predictions for post_ids: {', '.join(missing[:10])}")
    if extra:
        errors.append(f"Unexpected predictions for post_ids: {', '.join(extra[:10])}")

    return by_post_id, errors


def validate_prediction(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    task_a = parsed.get("task_a", {})
    task_b = parsed.get("task_b", {})

    a_id = str(task_a.get("label_id", "")).strip()
    b_id = str(task_b.get("label_id", "")).strip()

    if a_id not in TASK_A_LABELS:
        errors.append(f"Invalid Task A label_id: {a_id}")
        a_name = str(task_a.get("label_name", "")).strip()
    else:
        a_name = TASK_A_LABELS[a_id]

    if b_id not in TASK_B_LABELS:
        errors.append(f"Invalid Task B label_id: {b_id}")
        b_name = str(task_b.get("label_name", "")).strip()
    else:
        b_name = TASK_B_LABELS[b_id]

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("Invalid confidence value")

    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "task_a": {"label_id": a_id, "label_name": a_name},
        "task_b": {"label_id": b_id, "label_name": b_name},
        "confidence": confidence,
    }
    return normalized, errors


def call_ollama(
    messages: list[dict[str, str]],
    model: str,
    timeout: int,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "format": "json",
        "keep_alive": keep_alive,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[str] = []
            for line in response:
                if not line.strip():
                    continue
                data = json.loads(line.decode("utf-8"))
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                chunks.append(data.get("message", {}).get("content", ""))
                if data.get("done"):
                    break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    return "".join(chunks)


def call_openai(messages: list[dict[str, str]], model: str, timeout: int, temperature: float) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI package first: pip install openai") from exc

    client = OpenAI(timeout=timeout)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or ""


def load_completed_post_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = row.get("post_id")
            if post_id is not None:
                completed.add(str(post_id))
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jsonl_to_outputs(jsonl_path: Path, json_path: Path, csv_path: Path) -> None:
    rows = []
    fieldnames = [
        "experiment_id",
        "provider",
        "model",
        "shot",
        "text_source",
        "row_index",
        "post_id",
        "parse_ok",
        "task_a_label_id",
        "task_a_label_name",
        "task_b_label_id",
        "task_b_label_name",
        "confidence",
        "error",
        "created_at",
        "raw_response",
    ]

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def run_experiment(
    exp: Experiment,
    data: pd.DataFrame,
    output_dir: Path,
    resume: bool,
    limit: int | None,
    start_index: int | None,
    end_index: int | None,
    sleep_seconds: float,
    timeout: int,
    temperature: float,
    ollama_num_ctx: int,
    ollama_num_predict: int,
    ollama_keep_alive: str,
    max_retries: int,
    batch_size: int,
) -> None:
    exp_dir = output_dir / exp.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = exp_dir / "predictions.jsonl"
    json_path = exp_dir / "predictions.json"
    csv_path = exp_dir / "predictions.csv"
    metadata_path = exp_dir / "metadata.json"

    metadata = {
        "experiment_id": exp.experiment_id,
        "provider": exp.provider,
        "model": exp.model,
        "shot": exp.shot,
        "text_source": exp.text_source,
        "text_col": exp.text_col,
        "data_rows": int(len(data)),
        "batch_size": int(batch_size if exp.provider == "openai" else 1),
        "started_or_resumed_at": now_iso(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    completed = load_completed_post_ids(predictions_path) if resume else set()
    if completed:
        print(f"[{exp.experiment_id}] Resuming with {len(completed)} completed post_ids.")

    subset = data.copy()
    if start_index is not None:
        subset = subset[subset["_row_index"] >= start_index]
    if end_index is not None:
        subset = subset[subset["_row_index"] < end_index]
    if limit is not None:
        subset = subset.head(limit)

    total = len(subset)
    print(f"[{exp.experiment_id}] Rows selected for this run: {total}")

    pending_rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        post_id = str(row["post_id"])
        if post_id in completed:
            continue
        pending_rows.append(row.to_dict())

    if exp.provider == "openai":
        effective_batch_size = max(1, batch_size)
        print(f"[{exp.experiment_id}] OpenAI batch size: {effective_batch_size}")
        for batch_index, batch_rows in enumerate(chunks(pending_rows, effective_batch_size), start=1):
            nonempty_rows: list[dict[str, Any]] = []
            for row in batch_rows:
                post_id = str(row["post_id"])
                text = str(row.get(exp.text_col, "") or "").strip()
                if not text:
                    result = {
                        "experiment_id": exp.experiment_id,
                        "provider": exp.provider,
                        "model": exp.model,
                        "shot": exp.shot,
                        "text_source": exp.text_source,
                        "row_index": int(row["_row_index"]),
                        "post_id": post_id,
                        "parse_ok": False,
                        "task_a_label_id": "",
                        "task_a_label_name": "",
                        "task_b_label_id": "",
                        "task_b_label_name": "",
                        "confidence": "",
                        "error": f"Empty text column: {exp.text_col}",
                        "created_at": now_iso(),
                        "raw_response": "",
                    }
                    append_jsonl(predictions_path, result)
                    completed.add(post_id)
                else:
                    nonempty_rows.append(row)

            if not nonempty_rows:
                continue

            posts = [
                {
                    "post_id": str(row["post_id"]),
                    "text": str(row.get(exp.text_col, "") or "").strip(),
                }
                for row in nonempty_rows
            ]
            expected_post_ids = [post["post_id"] for post in posts]
            raw = ""
            error = ""
            by_post_id: dict[str, dict[str, Any]] = {}

            for attempt in range(1, max_retries + 1):
                try:
                    raw = call_openai(
                        make_batch_messages(posts, exp.shot),
                        exp.model,
                        timeout=timeout,
                        temperature=temperature,
                    )
                    parsed = extract_json_object(raw)
                    by_post_id, validation_errors = validate_batch_predictions(parsed, expected_post_ids)
                    if validation_errors:
                        raise ValueError("; ".join(validation_errors))
                    error = ""
                    break
                except Exception as exc:
                    error = f"Attempt {attempt}/{max_retries} failed: {type(exc).__name__}: {exc}"
                    print(f"[{exp.experiment_id}] batch {batch_index}: {error}", file=sys.stderr)
                    time.sleep(min(5 * attempt, 30))

            for row in nonempty_rows:
                post_id = str(row["post_id"])
                normalized = by_post_id.get(
                    post_id,
                    {
                        "task_a": {"label_id": "", "label_name": ""},
                        "task_b": {"label_id": "", "label_name": ""},
                        "confidence": "",
                    },
                )
                parsed_ok = post_id in by_post_id and not error
                result = {
                    "experiment_id": exp.experiment_id,
                    "provider": exp.provider,
                    "model": exp.model,
                    "shot": exp.shot,
                    "text_source": exp.text_source,
                    "row_index": int(row["_row_index"]),
                    "post_id": post_id,
                    "parse_ok": parsed_ok,
                    "task_a_label_id": normalized["task_a"]["label_id"],
                    "task_a_label_name": normalized["task_a"]["label_name"],
                    "task_b_label_id": normalized["task_b"]["label_id"],
                    "task_b_label_name": normalized["task_b"]["label_name"],
                    "confidence": normalized["confidence"],
                    "error": "" if parsed_ok else error,
                    "created_at": now_iso(),
                    "raw_response": raw,
                }
                append_jsonl(predictions_path, result)
                completed.add(post_id)

            processed = min(batch_index * effective_batch_size, len(pending_rows))
            print(f"[{exp.experiment_id}] processed {processed}/{len(pending_rows)} pending rows; completed file rows: {len(completed)}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    else:
        for n_done, row in enumerate(pending_rows, start=1):
            post_id = str(row["post_id"])
            text = str(row.get(exp.text_col, "") or "").strip()
            if not text:
                result = {
                    "experiment_id": exp.experiment_id,
                    "provider": exp.provider,
                    "model": exp.model,
                    "shot": exp.shot,
                    "text_source": exp.text_source,
                    "row_index": int(row["_row_index"]),
                    "post_id": post_id,
                    "parse_ok": False,
                    "task_a_label_id": "",
                    "task_a_label_name": "",
                    "task_b_label_id": "",
                    "task_b_label_name": "",
                    "confidence": "",
                    "error": f"Empty text column: {exp.text_col}",
                    "created_at": now_iso(),
                    "raw_response": "",
                }
                append_jsonl(predictions_path, result)
                completed.add(post_id)
                continue

            messages = make_messages(text, exp.shot)
            raw = ""
            error = ""
            parsed_ok = False
            normalized = {
                "task_a": {"label_id": "", "label_name": ""},
                "task_b": {"label_id": "", "label_name": ""},
                "confidence": "",
            }

            for attempt in range(1, max_retries + 1):
                try:
                    raw = call_ollama(
                        messages,
                        exp.model,
                        timeout=timeout,
                        temperature=temperature,
                        num_ctx=ollama_num_ctx,
                        num_predict=ollama_num_predict,
                        keep_alive=ollama_keep_alive,
                    )

                    parsed = extract_json_object(raw)
                    normalized, validation_errors = validate_prediction(parsed)
                    if validation_errors:
                        error = "; ".join(validation_errors)
                    else:
                        error = ""
                        parsed_ok = True
                    break
                except Exception as exc:
                    error = f"Attempt {attempt}/{max_retries} failed: {type(exc).__name__}: {exc}"
                    print(f"[{exp.experiment_id}] {post_id}: {error}", file=sys.stderr)
                    time.sleep(min(5 * attempt, 30))

            result = {
                "experiment_id": exp.experiment_id,
                "provider": exp.provider,
                "model": exp.model,
                "shot": exp.shot,
                "text_source": exp.text_source,
                "row_index": int(row["_row_index"]),
                "post_id": post_id,
                "parse_ok": parsed_ok,
                "task_a_label_id": normalized["task_a"]["label_id"],
                "task_a_label_name": normalized["task_a"]["label_name"],
                "task_b_label_id": normalized["task_b"]["label_id"],
                "task_b_label_name": normalized["task_b"]["label_name"],
                "confidence": normalized["confidence"],
                "error": error,
                "created_at": now_iso(),
                "raw_response": raw,
            }
            append_jsonl(predictions_path, result)
            completed.add(post_id)

            if n_done % 25 == 0:
                print(f"[{exp.experiment_id}] processed {n_done}/{len(pending_rows)} pending rows; completed file rows: {len(completed)}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    jsonl_to_outputs(predictions_path, json_path, csv_path)
    print(f"[{exp.experiment_id}] Wrote {predictions_path}")
    print(f"[{exp.experiment_id}] Wrote {json_path}")
    print(f"[{exp.experiment_id}] Wrote {csv_path}")


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["post_id", "native_text", "translated_text"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    data = data.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    data["_row_index"] = data.index
    for col in ["native_text", "translated_text"]:
        data[col] = data[col].fillna("").astype(str)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AquaCrisis LLM classification experiments.")
    parser.add_argument("--data", required=True, type=Path, help="Input CSV with post_id, native_clean, translated_clean.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for experiment outputs.")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENT_BY_ID), help="Experiment ID. Can be repeated.")
    parser.add_argument("--provider", choices=["ollama", "openai"], help="Run all experiments for this provider.")
    parser.add_argument("--resume", action="store_true", help="Skip post_ids already present in predictions.jsonl.")
    parser.add_argument("--limit", type=int, help="Optional limit for test runs.")
    parser.add_argument("--start-index", type=int, help="Optional inclusive row start index.")
    parser.add_argument("--end-index", type=int, help="Optional exclusive row end index.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep after each request.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP/API timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature.")
    parser.add_argument("--ollama-num-ctx", type=int, default=4096, help="Ollama context window for local models.")
    parser.add_argument("--ollama-num-predict", type=int, default=256, help="Maximum generated tokens for Ollama calls.")
    parser.add_argument("--ollama-keep-alive", default="30m", help="How long Ollama should keep the model loaded.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row.")
    parser.add_argument("--batch-size", type=int, default=20, help="Posts per OpenAI API request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment:
        experiments = [EXPERIMENT_BY_ID[name] for name in args.experiment]
    elif args.provider:
        experiments = [exp for exp in EXPERIMENTS if exp.provider == args.provider]
    else:
        experiments = EXPERIMENTS

    print("Selected experiments:")
    for exp in experiments:
        print(f"- {exp.experiment_id}: {exp.provider} | {exp.model} | {exp.shot} | {exp.text_source}")

    for exp in experiments:
        run_experiment(
            exp=exp,
            data=data,
            output_dir=args.output_dir,
            resume=args.resume,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            temperature=args.temperature,
            ollama_num_ctx=args.ollama_num_ctx,
            ollama_num_predict=args.ollama_num_predict,
            ollama_keep_alive=args.ollama_keep_alive,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
