from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

TARGET_COLUMN = "Class"
SPLITS = ("train", "val", "test")
RAW_FILE_TEMPLATE = "Dry_Bean_Dataset_Dirty_{split}.csv"
CLEAN_FILE_TEMPLATE = "dry_bean_clean_{split}.csv"

CANONICAL_CLASSES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

LABEL_FIXES = {
    "BARBUNYA": "BARBUNYA",
    "BOMBAY": "BOMBAY",
    "CALI": "CALI",
    "DERMASON": "DERMASON",
    "HOROZ": "HOROZ",
    "SEKER": "SEKER",
    "SIRA": "SIRA",
    "D3RMAS0N": "DERMASON",
    "S3K3R": "SEKER",
    "H0R0Z": "HOROZ",
    "B0MBAY": "BOMBAY",
}


def ensure_dirs() -> None:
    for path in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, METRICS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({(key or "").strip(): value for key, value in row.items()})
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def numeric_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name != TARGET_COLUMN]


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if parsed < 0:
        return None
    return parsed


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return LABEL_FIXES.get(text)


def median_by_column(rows: list[dict[str, object]], columns: list[str]) -> dict[str, float]:
    medians = {}
    for col in columns:
        values = [float(row[col]) for row in rows if row.get(col) is not None]
        medians[col] = float(median(values))
    return medians


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
