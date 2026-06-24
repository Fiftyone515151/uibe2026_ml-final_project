from __future__ import annotations

from collections import Counter, defaultdict

from common import (
    CANONICAL_CLASSES,
    CLEAN_FILE_TEMPLATE,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILE_TEMPLATE,
    REPORTS_DIR,
    SPLITS,
    TARGET_COLUMN,
    ensure_dirs,
    median_by_column,
    normalize_label,
    numeric_columns,
    parse_float,
    read_csv,
    rounded,
    write_csv,
    write_json,
)


def profile_raw_rows(rows: list[dict[str, str]], features: list[str]) -> dict[str, object]:
    missing = Counter()
    invalid_numeric = Counter()
    negative_numeric = Counter()
    labels = Counter()

    for row in rows:
        labels[row.get(TARGET_COLUMN, "")] += 1
        for col in features:
            raw = row.get(col)
            if raw is None or not str(raw).strip():
                missing[col] += 1
                continue
            try:
                value = float(str(raw).strip())
            except ValueError:
                invalid_numeric[col] += 1
                continue
            if value < 0:
                negative_numeric[col] += 1

    return {
        "rows": len(rows),
        "raw_class_counts": dict(labels),
        "missing_values": dict(missing),
        "invalid_numeric_values": dict(invalid_numeric),
        "negative_numeric_values": dict(negative_numeric),
    }


def clean_split(
    rows: list[dict[str, str]],
    features: list[str],
    split: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cleaned = []
    dropped_unknown_label = 0
    label_changes = Counter()
    invalid_by_column = Counter()
    duplicate_count = 0
    seen = set()

    for row in rows:
        raw_label = row.get(TARGET_COLUMN)
        label = normalize_label(raw_label)
        if label is None:
            dropped_unknown_label += 1
            continue
        raw_label_clean = str(raw_label or "").strip()
        if raw_label_clean != label:
            label_changes[f"{raw_label_clean} -> {label}"] += 1

        out = {}
        for col in features:
            value = parse_float(row.get(col))
            if value is None:
                invalid_by_column[col] += 1
            out[col] = value
        out[TARGET_COLUMN] = label

        key = tuple(out[col] for col in features + [TARGET_COLUMN])
        if split == "train" and key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        cleaned.append(out)

    return cleaned, {
        "rows_after_label_filter_and_duplicate_removal": len(cleaned),
        "dropped_unknown_label": dropped_unknown_label,
        "duplicates_removed": duplicate_count,
        "label_changes": dict(label_changes),
        "invalid_or_missing_feature_values_after_parsing": dict(invalid_by_column),
    }


def impute_rows(
    rows: list[dict[str, object]],
    features: list[str],
    medians: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    imputed = Counter()
    out_rows = []
    for row in rows:
        out = dict(row)
        for col in features:
            if out[col] is None:
                out[col] = medians[col]
                imputed[col] += 1
            else:
                out[col] = rounded(float(out[col]))
        out[TARGET_COLUMN] = str(out[TARGET_COLUMN])
        out_rows.append(out)
    return out_rows, dict(imputed)


def class_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(row[TARGET_COLUMN]) for row in rows)
    return {name: counts.get(name, 0) for name in CANONICAL_CLASSES}


def main() -> None:
    ensure_dirs()
    raw_payload = {}
    cleaned_payload = {}
    fieldnames = None
    features = None

    for split in SPLITS:
        current_fieldnames, rows = read_csv(RAW_DIR / RAW_FILE_TEMPLATE.format(split=split))
        if fieldnames is None:
            fieldnames = current_fieldnames
            features = numeric_columns(fieldnames)
        assert features is not None
        raw_payload[split] = {
            "file": RAW_FILE_TEMPLATE.format(split=split),
            **profile_raw_rows(rows, features),
        }
        cleaned_rows, split_report = clean_split(rows, features, split)
        cleaned_payload[split] = {
            "rows": cleaned_rows,
            "report": split_report,
        }

    assert features is not None and fieldnames is not None
    train_rows = cleaned_payload["train"]["rows"]
    medians = median_by_column(train_rows, features)

    final_reports = {}
    for split in SPLITS:
        rows = cleaned_payload[split]["rows"]
        final_rows, imputed = impute_rows(rows, features, medians)
        cleaned_payload[split]["rows"] = final_rows
        output_name = CLEAN_FILE_TEMPLATE.format(split=split)
        write_csv(PROCESSED_DIR / output_name, fieldnames, final_rows)
        final_reports[split] = {
            **cleaned_payload[split]["report"],
            "output_file": output_name,
            "rows_final": len(final_rows),
            "class_counts_final": class_counts(final_rows),
            "imputed_values": imputed,
        }

    report = {
        "raw_profile": raw_payload,
        "cleaning_rules": [
            "strip whitespace from labels and column names",
            "normalize class labels to seven canonical bean classes",
            "convert feature columns to non-negative finite floats",
            "treat missing, non-numeric, non-finite, and negative feature values as missing",
            "remove duplicate rows from training split",
            "impute missing feature values using training-set medians",
        ],
        "training_medians_used_for_imputation": {k: rounded(v) for k, v in medians.items()},
        "cleaned_profile": final_reports,
    }
    write_json(REPORTS_DIR / "cleaning_report.json", report)

    print("Generated cleaned datasets and cleaning_report.json")


if __name__ == "__main__":
    main()
