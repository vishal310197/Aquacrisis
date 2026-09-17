#!/usr/bin/env python3
"""
Build a Facebook Post Table and translate non-English posts to English.
Designed for CSC Mahti / Slurm batch jobs.

Outputs:
  - post_table.csv
  - post_table.xlsx
  - translation_cache.jsonl
  - run_summary.json

Important:
  Do NOT hard-code your OpenAI key in this file.
  Export it in the shell or Slurm script as OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder


FINAL_COLUMNS = [
    "post_id",
    "utility_id",
    "event_id",
    "created_at",
    "language",
    "native_text",
    "translated_text",
    "reaction_count",
    "comment_count",
    "share_count",
    "media_type",
    "url",
]

# Edit these IDs if your Utility Table uses a different order.
UTILITY_URL_TO_ID = {
    "https://www.facebook.com/helsinginseudunymparistopalvelut": "U01",  # HSY
    "https://www.facebook.com/HamburgWasser": "U02",
    "https://www.facebook.com/EPALaguaslivres": "U03",
    "https://www.facebook.com/Oslovann": "U04",
    "https://www.facebook.com/wasserbetriebe": "U05",
    "https://www.facebook.com/worldwaternetnl": "U06",
    "https://www.facebook.com/FARYSmultiservice": "U07",
    "https://www.facebook.com/eaudeparis": "U08",
    "https://www.facebook.com/abcacquabenecomunenapoli": "U09",
    "https://www.facebook.com/stockholmvattenochavfall": "U10",
    "https://www.facebook.com/tampereenvesi": "U11",
    "https://www.facebook.com/turunvesihuolto": "U12",
}

UTILITY_NAME_TO_ID = {
    "hsy": "U01",
    "helsingin seudun ympäristöpalvelut": "U01",
    "hamburg wasser": "U02",
    "epal": "U03",
    "oslo": "U04",
    "berliner wasserbetriebe": "U05",
    "waternet": "U06",
    "world waternet": "U06",
    "farys": "U07",
    "eau de paris": "U08",
    "abc napoli": "U09",
    "stockholm vatten och avfall": "U10",
    "tampereen vesi": "U11",
    "turun vesihuolto": "U12",
}

# Optional first-pass incident linking rules. Keep unmatched rows blank.
INCIDENT_RULES = [
    {
        "event_id": "E2025_STO_SAVE_WATER_01",
        "utility_id": "U10",
        "start": "2025-08-15",
        "end": "2025-08-31",
        "keywords": [
            "spara vatten",
            "använd mindre vatten",
            "vattenförbrukning",
            "vattenverk",
            "ansträngd situation",
            "normal vattenförbrukning",
        ],
    },
    {
        "event_id": "E2026_HSY_VALLILA_BWA_01",
        "utility_id": "U01",
        "start": "2026-03-31",
        "end": "2026-04-02",
        "keywords": [
            "kokningsrekommendation",
            "koka dricksvatten",
            "vallila",
            "boil-water",
            "boil water",
        ],
    },
    {
        "event_id": "E2024_ABC_NAPOLI_SHUTOFF_01",
        "utility_id": "U09",
        "start": "2024-12-04",
        "end": "2024-12-04",
        "keywords": [
            "interruzione idrica",
            "sospensione",
            "viale giochi del mediterraneo",
            "lavori programmati",
        ],
    },
    {
        "event_id": "E2026_ABC_NAPOLI_SHUTOFF_01",
        "utility_id": "U09",
        "start": "2026-01-29",
        "end": "2026-01-29",
        "keywords": ["interruzione idrica programmata", "10:00", "18:00"],
    },
]


def normalize_fb_url(url: Any) -> str:
    if pd.isna(url):
        return ""
    url = str(url).strip().split("?")[0].rstrip("/")
    return url


def clean_count(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return pd.NA
    try:
        return int(float(value))
    except Exception:
        return pd.NA


def clean_post_id(value: Any, url: Any = None) -> Any:
    if not pd.isna(value):
        value = str(value).strip()
        value = re.sub(r"\.0$", "", value)
        if value:
            return value
    if not pd.isna(url):
        url = str(url)
        match = re.search(r"(pfbid[0-9A-Za-z]+|/reel/([0-9]+)|/posts/([^/?]+))", url)
        if match:
            return match.group(1).strip("/")
    return pd.NA


def get_column(df: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def get_utility_id(row: pd.Series) -> Any:
    input_url = normalize_fb_url(row.get("inputUrl", ""))
    for known_url, utility_id in UTILITY_URL_TO_ID.items():
        if normalize_fb_url(known_url).lower() == input_url.lower():
            return utility_id

    page_name = str(row.get("user/name", "")).strip().lower()
    for key, utility_id in UTILITY_NAME_TO_ID.items():
        if key in page_name:
            return utility_id
    return pd.NA


def parse_created_at(series: pd.Series) -> pd.Series:
    """Parse Facebook createdAt whether it is Unix seconds/ms or an ISO datetime string."""
    numeric = pd.to_numeric(series, errors="coerce")
    # If most non-empty values are numeric, assume seconds unless values look like ms.
    if numeric.notna().sum() > max(1, len(series) * 0.5):
        median_val = numeric.dropna().median()
        unit = "ms" if median_val > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def detect_media_type(row: pd.Series) -> Any:
    url = str(row.get("url", "")).lower()
    att_type_0 = str(row.get("attachments/0/type", "")).lower()
    att_type_1 = str(row.get("attachments/1/type", "")).lower()
    combined = " ".join([url, att_type_0, att_type_1])

    if "reel" in url or "video" in combined:
        return "video"
    if "photo" in combined or "album" in combined:
        return "image"
    for col in [
        "attachments/0/media/image/uri",
        "attachments/0/media/image/src",
        "attachments/1/media/image/uri",
        "attachments/1/media/image/src",
    ]:
        if col in row.index and not pd.isna(row.get(col)):
            return "image"
    for col in [
        "attachments/0/url",
        "attachments/0/target/url",
        "attachments/0/media/permalink_url",
        "attachments/1/url",
        "attachments/1/target/url",
    ]:
        if col in row.index and not pd.isna(row.get(col)):
            candidate = str(row.get(col)).lower()
            if candidate.startswith("http") and "facebook.com/reel" not in candidate:
                return "link"
    if str(row.get("text", "")).strip():
        return "text"
    return pd.NA


def link_event_id(row: pd.Series) -> Any:
    utility_id = row.get("utility_id")
    text = str(row.get("native_text", "")).lower()
    created_at = row.get("created_at_dt")
    if pd.isna(utility_id) or pd.isna(created_at):
        return pd.NA

    for rule in INCIDENT_RULES:
        if utility_id != rule["utility_id"]:
            continue
        start = pd.to_datetime(rule["start"], utc=True)
        end = pd.to_datetime(rule["end"], utc=True) + pd.Timedelta(days=1)
        if start <= created_at < end and any(k.lower() in text for k in rule["keywords"]):
            return rule["event_id"]
    return pd.NA


def build_language_detector():
    # Lingua enum names vary by version. This avoids AttributeError.
    candidate_language_names = [
        "ENGLISH",
        "DUTCH",
        "SWEDISH",
        "FINNISH",
        "FRENCH",
        "GERMAN",
        "ITALIAN",
        "PORTUGUESE",
        "NORWEGIAN",
        "BOKMAL",
        "NYNORSK",
        "DANISH",
        "SPANISH",
    ]
    languages = [getattr(Language, name) for name in candidate_language_names if hasattr(Language, name)]
    if not languages:
        raise RuntimeError("No Lingua languages could be loaded. Check lingua-language-detector installation.")
    return LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().build()


LANG_NAME_FIX = {
    "BOKMAL": "Norwegian",
    "NYNORSK": "Norwegian",
    "NORWEGIAN": "Norwegian",
    "NORWEGIAN_BOKMAL": "Norwegian",
    "NORWEGIAN_NYNORSK": "Norwegian",
}


def detect_language_name(detector, text: Any) -> Any:
    text = str(text).strip()
    if not text:
        return pd.NA
    detected = detector.detect_language_of(text)
    if detected is None:
        return pd.NA
    raw = detected.name
    name = LANG_NAME_FIX.get(raw, raw)
    return name.replace("_", " ").title()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("hash") and "translated_text" in obj:
                    cache[obj["hash"]] = obj.get("translated_text", "")
            except json.JSONDecodeError:
                continue
    return cache


def append_cache(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class TranslationError(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    retry=retry_if_exception_type((TranslationError, Exception)),
)
def translate_batch_openai(client: OpenAI, model: str, records: List[Dict[str, str]]) -> Dict[str, str]:
    system_prompt = """
You are a professional translation engine for public-sector social media posts.
Translate each text into clear English.

Rules:
- Preserve numbers, dates, street names, place names, hashtags, URLs, and emojis.
- Do not summarize.
- Do not add information.
- If the text is already English, return it unchanged.
- If text is empty, return an empty string.
- Return ONLY valid JSON as a list of objects:
  [{"key": "...", "translated_text": "..."}]
""".strip()

    payload = [
        {"key": str(r["key"]), "language": r.get("language", ""), "text": r.get("text", "")}
        for r in records
    ]

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
    )

    raw = response.output_text.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            raise TranslationError(f"Model did not return JSON. First 500 chars: {raw[:500]}")
        parsed = json.loads(match.group(0))

    result = {str(item.get("key")): item.get("translated_text", "") for item in parsed}
    missing = [r["key"] for r in payload if str(r["key"]) not in result]
    if missing:
        raise TranslationError(f"Translation response missing keys: {missing[:10]}")
    return result




def log_translation_failure(path: Path, batch: List[Dict[str, str]], error: Exception) -> None:
    """Append failed translation records to a JSONL file for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    err = repr(error)
    with path.open("a", encoding="utf-8") as f:
        for item in batch:
            f.write(json.dumps({
                "key": str(item.get("key", "")),
                "hash": item.get("hash", ""),
                "language": item.get("language", ""),
                "native_text": item.get("text", ""),
                "error": err,
                "created_at_unix": int(time.time()),
            }, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def translate_batch_resilient(
    client: OpenAI,
    model: str,
    batch: List[Dict[str, str]],
    failure_log_path: Path,
) -> Dict[str, str]:
    """
    Translate a batch robustly.

    If a large batch fails because the LLM returns malformed JSON or misses keys,
    split the batch recursively. This preserves progress and prevents one noisy
    Facebook post from crashing the whole Slurm job.
    """
    try:
        return translate_batch_openai(client, model, batch)
    except Exception as exc:
        print(
            f"WARNING: batch of {len(batch)} failed after retries: {repr(exc)}",
            flush=True,
        )

        if len(batch) <= 1:
            log_translation_failure(failure_log_path, batch, exc)
            # Keep the pipeline moving. Failed singleton gets blank translation.
            return {str(batch[0].get("key")): ""}

        mid = len(batch) // 2
        left = translate_batch_resilient(client, model, batch[:mid], failure_log_path)
        right = translate_batch_resilient(client, model, batch[mid:], failure_log_path)
        left.update(right)
        return left


# Excel/openpyxl cannot write ASCII control characters such as NUL, , , etc.
# Facebook text can contain these hidden characters, so remove them before XLSX export.
EXCEL_ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_for_excel(value: Any) -> Any:
    """Remove characters that openpyxl rejects from strings."""
    if isinstance(value, str):
        # Also limit cell text to Excel's 32,767 character cell limit.
        value = EXCEL_ILLEGAL_CHARACTERS_RE.sub("", value)
        if len(value) > 32767:
            value = value[:32760] + " [TRUNCATED]"
    return value


def export_outputs(post_table: pd.DataFrame, output_dir: Path, csv_name: str, xlsx_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = post_table[FINAL_COLUMNS].copy()
    out = out.replace({pd.NA: "", "nan": ""})

    csv_path = output_dir / csv_name
    xlsx_path = output_dir / xlsx_name

    # CSV can safely keep original Unicode text, but remove NUL/control chars too
    # so spreadsheet programs do not break when opening the CSV.
    out_csv = out.applymap(clean_for_excel)
    out_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")

    out_xlsx = out.applymap(clean_for_excel)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        out_xlsx.to_excel(writer, index=False, sheet_name="Post Table")
        ws = writer.book["Post Table"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        widths = {
            "A": 26,
            "B": 12,
            "C": 30,
            "D": 22,
            "E": 16,
            "F": 65,
            "G": 65,
            "H": 16,
            "I": 16,
            "J": 14,
            "K": 14,
            "L": 75,
        }
        from openpyxl.styles import Alignment, Font

        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cell in ws[1]:
            cell.font = Font(bold=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and translate Facebook Post Table.")
    parser.add_argument("--input", required=True, help="Input Facebook posts CSV")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--output-csv", default="post_table.csv")
    parser.add_argument("--output-xlsx", default="post_table.xlsx")
    parser.add_argument("--cache", default="translation_cache.jsonl", help="Translation cache JSONL path")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI translation model")
    parser.add_argument("--batch-size", type=int, default=100, help="Unique texts per API call")
    parser.add_argument("--max-unique-to-translate", type=int, default=None, help="Limit unique translations for test runs")
    parser.add_argument("--no-translate", action="store_true", help="Build table and detect language but do not call OpenAI")
    parser.add_argument("--partial-every", type=int, default=10, help="Export partial table every N batches")
    parser.add_argument("--sleep-between-batches", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = output_dir / cache_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input CSV: {input_path}", flush=True)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Rows loaded: {len(df):,}", flush=True)
    print(f"Columns loaded: {len(df.columns):,}", flush=True)

    post_table = pd.DataFrame(index=df.index)
    post_table["post_id"] = df.apply(lambda r: clean_post_id(r.get("id"), r.get("url")), axis=1)
    post_table["utility_id"] = df.apply(get_utility_id, axis=1)

    created_at_dt = parse_created_at(get_column(df, "createdAt"))
    post_table["created_at_dt"] = created_at_dt
    # Format as ISO-ish UTC, e.g. 2025-08-15 12:34:56+0000
    post_table["created_at"] = created_at_dt.dt.strftime("%Y-%m-%d %H:%M:%S%z")

    post_table["native_text"] = get_column(df, "text", "").fillna("").astype(str).str.strip()
    post_table["reaction_count"] = get_column(df, "reactionCount").apply(clean_count)
    post_table["comment_count"] = get_column(df, "commentCount").apply(clean_count)
    post_table["share_count"] = get_column(df, "shareCount").apply(clean_count)
    post_table["media_type"] = df.apply(detect_media_type, axis=1)
    post_table["url"] = get_column(df, "url", "")
    post_table["event_id"] = post_table.apply(link_event_id, axis=1)

    print("Detecting language locally with Lingua...", flush=True)
    detector = build_language_detector()
    post_table["language"] = [
        detect_language_name(detector, text)
        for text in tqdm(post_table["native_text"], desc="Language detection")
    ]

    post_table["translated_text"] = ""

    needs_translation = (
        post_table["native_text"].fillna("").str.strip().ne("")
        & post_table["language"].fillna("").str.lower().ne("english")
    )

    # English rows are copied unchanged.
    english_rows = post_table["native_text"].fillna("").str.strip().ne("") & post_table[
        "language"
    ].fillna("").str.lower().eq("english")
    post_table.loc[english_rows, "translated_text"] = post_table.loc[english_rows, "native_text"]

    unique_items = (
        post_table.loc[needs_translation, ["native_text", "language"]]
        .drop_duplicates(subset=["native_text"])
        .reset_index(drop=True)
    )

    print(f"Non-empty posts: {post_table['native_text'].str.strip().ne('').sum():,}", flush=True)
    print(f"Rows needing translation: {needs_translation.sum():,}", flush=True)
    print(f"Unique non-English texts: {len(unique_items):,}", flush=True)

    if args.max_unique_to_translate is not None:
        unique_items = unique_items.head(args.max_unique_to_translate).copy()
        print(f"TEST LIMIT active: translating only first {len(unique_items):,} unique texts", flush=True)

    cache = load_cache(cache_path)
    print(f"Loaded cached translations: {len(cache):,}", flush=True)

    # Prepare records not already in cache.
    records: List[Dict[str, str]] = []
    for i, row in unique_items.iterrows():
        text = str(row["native_text"]).strip()
        h = text_hash(text)
        if h in cache:
            continue
        records.append({"key": str(i), "hash": h, "language": str(row["language"]), "text": text})

    print(f"Unique texts still to translate: {len(records):,}", flush=True)

    if args.no_translate:
        print("--no-translate used: skipping OpenAI calls.", flush=True)
    elif records:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Export it before running the job.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        batches = [records[i : i + args.batch_size] for i in range(0, len(records), args.batch_size)]
        print(f"Translating {len(records):,} unique texts in {len(batches):,} batches", flush=True)
        print(f"Model: {args.model}; batch size: {args.batch_size}", flush=True)

        for batch_idx, batch in enumerate(tqdm(batches, desc="Translating batches"), start=1):
            failure_log_path = output_dir / "translation_failures.jsonl"
            batch_result = translate_batch_resilient(client, args.model, batch, failure_log_path)
            cache_items = []
            for item in batch:
                translated = batch_result.get(str(item["key"]), "")
                cache[item["hash"]] = translated
                cache_items.append(
                    {
                        "hash": item["hash"],
                        "language": item["language"],
                        "native_text": item["text"],
                        "translated_text": translated,
                        "model": args.model,
                        "created_at_unix": int(time.time()),
                    }
                )
            append_cache(cache_path, cache_items)

            if args.partial_every and batch_idx % args.partial_every == 0:
                print(f"Writing partial outputs after batch {batch_idx}/{len(batches)}", flush=True)
                # Fill currently available translations before partial export.
                tmp = post_table.copy()
                tmp["translated_text"] = tmp.apply(
                    lambda r: r["native_text"]
                    if str(r.get("language", "")).lower() == "english"
                    else cache.get(text_hash(str(r["native_text"]).strip()), ""),
                    axis=1,
                )
                export_outputs(tmp, output_dir, "post_table_partial.csv", "post_table_partial.xlsx")

            if args.sleep_between_batches > 0:
                time.sleep(args.sleep_between_batches)

    # Fill final translations from cache.
    def fill_translation(row: pd.Series) -> str:
        text = str(row.get("native_text", "")).strip()
        if not text:
            return ""
        if str(row.get("language", "")).strip().lower() == "english":
            return text
        return cache.get(text_hash(text), "")

    post_table["translated_text"] = post_table.apply(fill_translation, axis=1)
    missing_final = (
        needs_translation
        & post_table["translated_text"].fillna("").str.strip().eq("")
    ).sum()

    post_table_final = post_table[FINAL_COLUMNS].copy()
    post_table_final = post_table_final[
        post_table_final["post_id"].astype(str).str.strip().ne("")
        | post_table_final["native_text"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)

    print("Exporting final CSV and Excel...", flush=True)
    export_outputs(post_table_final, output_dir, args.output_csv, args.output_xlsx)

    summary = {
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "rows_loaded": int(len(df)),
        "rows_exported": int(len(post_table_final)),
        "rows_needing_translation": int(needs_translation.sum()),
        "unique_non_english_texts": int(len(unique_items)),
        "cached_translations": int(len(cache)),
        "missing_translations_after_run": int(missing_final),
        "model": args.model,
        "batch_size": int(args.batch_size),
        "runtime_seconds": round(time.time() - started, 2),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
