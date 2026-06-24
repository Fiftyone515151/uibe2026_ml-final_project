from __future__ import annotations

import math
import subprocess
from collections import Counter
from statistics import mean, median, pstdev

from common import (
    CANONICAL_CLASSES,
    CLEAN_FILE_TEMPLATE,
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLITS,
    TARGET_COLUMN,
    ensure_dirs,
    numeric_columns,
    read_csv,
    read_json,
    rounded,
    write_json,
)


PALETTE = [
    "#28666e",
    "#7c3aed",
    "#d97706",
    "#0f766e",
    "#be123c",
    "#2563eb",
    "#65a30d",
    "#9333ea",
]


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path, width: int, height: int, body: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}'
        '.title{font-size:22px;font-weight:700}.axis{font-size:12px;fill:#53616f}'
        '.label{font-size:13px;fill:#172033}.small{font-size:11px;fill:#53616f}</style>\n'
        f"{body}\n</svg>\n",
        encoding="utf-8",
    )


def svg_to_png(svg_path) -> str:
    png_path = svg_path.with_suffix(".png")
    subprocess.run(
        ["rsvg-convert", "-w", "1600", "-f", "png", "-o", str(png_path), str(svg_path)],
        check=True,
    )
    return png_path.name


def save_figure(svg_path, width: int, height: int, body: str) -> str:
    write_svg(svg_path, width, height, body)
    return svg_to_png(svg_path)


def bar_chart(path, title: str, data: dict[str, int], y_label: str = "Count") -> None:
    width, height = 980, 560
    left, right, top, bottom = 92, 36, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    labels = list(data.keys())
    values = [data[k] for k in labels]
    max_value = max(values) if values else 1
    step = plot_w / max(len(labels), 1)
    bar_w = step * 0.62
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<text x="22" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 22 {top + plot_h / 2})">{y_label}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    for i in range(6):
        value = max_value * i / 5
        y = top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{int(value)}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * step + (step - bar_w) / 2
        h = 0 if max_value == 0 else value / max_value * plot_h
        y = top + plot_h - h
        color = PALETTE[idx % len(PALETTE)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="small">{value}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{svg_escape(label)}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def grouped_bar_chart(path, title: str, split_counts: dict[str, dict[str, int]]) -> str:
    width, height = 1080, 600
    left, right, top, bottom = 92, 160, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    classes = CANONICAL_CLASSES
    splits = list(split_counts.keys())
    max_value = max(max(counts.values()) for counts in split_counts.values())
    group_step = plot_w / len(classes)
    bar_w = group_step / (len(splits) + 1)
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    for i in range(6):
        value = max_value * i / 5
        y = top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{int(value)}</text>')
    for ci, cls in enumerate(classes):
        base_x = left + ci * group_step
        for si, split in enumerate(splits):
            value = split_counts[split].get(cls, 0)
            h = value / max_value * plot_h
            x = base_x + si * bar_w + bar_w * 0.45
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.78:.1f}" height="{h:.1f}" fill="{PALETTE[si]}" rx="3"/>')
        parts.append(f'<text x="{base_x + group_step / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{cls}</text>')
    for si, split in enumerate(splits):
        y = top + si * 24
        parts.append(f'<rect x="{width - right + 30}" y="{y - 12}" width="14" height="14" fill="{PALETTE[si]}" rx="2"/>')
        parts.append(f'<text x="{width - right + 52}" y="{y}" class="label">{split}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def compact_grouped_bar_chart(path, title: str, groups: dict[str, dict[str, int]]) -> str:
    width, height = 1080, 600
    left, right, top, bottom = 92, 180, 70, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    group_names = list(groups.keys())
    series_names = list(next(iter(groups.values())).keys()) if groups else []
    max_value = max(max(values.values()) for values in groups.values()) if groups else 1
    group_step = plot_w / max(len(group_names), 1)
    bar_w = group_step / (len(series_names) + 1)
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    for i in range(6):
        value = max_value * i / 5
        y = top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{int(value)}</text>')
    for gi, group in enumerate(group_names):
        base_x = left + gi * group_step
        for si, series in enumerate(series_names):
            value = groups[group].get(series, 0)
            h = 0 if max_value == 0 else value / max_value * plot_h
            x = base_x + si * bar_w + bar_w * 0.45
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.78:.1f}" height="{h:.1f}" fill="{PALETTE[si]}" rx="3"/>')
            parts.append(f'<text x="{x + bar_w * 0.39:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="small">{value}</text>')
        parts.append(f'<text x="{base_x + group_step / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" class="axis">{group}</text>')
    for si, series in enumerate(series_names):
        y = top + si * 24
        parts.append(f'<rect x="{width - right + 28}" y="{y - 12}" width="14" height="14" fill="{PALETTE[si]}" rx="2"/>')
        parts.append(f'<text x="{width - right + 50}" y="{y}" class="label">{svg_escape(series)}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def histogram(path, title: str, values: list[float], bins: int = 20) -> str:
    width, height = 980, 560
    left, right, top, bottom = 86, 36, 70, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
        counts[idx] += 1
    max_count = max(counts)
    bar_w = plot_w / bins
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    for i, count in enumerate(counts):
        h = count / max_count * plot_h
        x = left + i * bar_w
        y = top + plot_h - h
        parts.append(f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="#28666e"/>')
    parts.append(f'<text x="{left}" y="{top + plot_h + 32}" class="axis">{lo:.2f}</text>')
    parts.append(f'<text x="{width - right}" y="{top + plot_h + 32}" text-anchor="end" class="axis">{hi:.2f}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def correlation_matrix(rows: list[dict[str, str]], features: list[str]) -> dict[str, dict[str, float]]:
    columns = {col: [float(row[col]) for row in rows] for col in features}
    means = {col: mean(vals) for col, vals in columns.items()}
    corr = {}
    for a in features:
        corr[a] = {}
        for b in features:
            num = sum((x - means[a]) * (y - means[b]) for x, y in zip(columns[a], columns[b]))
            den_a = math.sqrt(sum((x - means[a]) ** 2 for x in columns[a]))
            den_b = math.sqrt(sum((y - means[b]) ** 2 for y in columns[b]))
            corr[a][b] = 0.0 if den_a == 0 or den_b == 0 else num / (den_a * den_b)
    return corr


def heatmap(path, title: str, corr: dict[str, dict[str, float]], features: list[str]) -> str:
    cell = 38
    left, top = 190, 76
    width = left + cell * len(features) + 36
    height = top + cell * len(features) + 190
    parts = [f'<text x="36" y="36" class="title">{svg_escape(title)}</text>']
    for i, row_name in enumerate(features):
        y = top + i * cell
        parts.append(f'<text x="{left - 8}" y="{y + 24}" text-anchor="end" class="axis">{svg_escape(row_name)}</text>')
        parts.append(f'<text x="{left + i * cell + 20}" y="{top + cell * len(features) + 18}" text-anchor="end" class="axis" transform="rotate(-45 {left + i * cell + 20} {top + cell * len(features) + 18})">{svg_escape(row_name)}</text>')
        for j, col_name in enumerate(features):
            value = corr[row_name][col_name]
            if value >= 0:
                intensity = int(255 - 155 * abs(value))
                color = f"rgb({intensity},{intensity + 15},255)"
            else:
                intensity = int(255 - 155 * abs(value))
                color = f"rgb(255,{intensity + 8},{intensity})"
            x = left + j * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
    return save_figure(path, width, height, "\n".join(parts))


def load_cleaned() -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    split_rows = {}
    fieldnames = None
    for split in SPLITS:
        current_fieldnames, rows = read_csv(PROCESSED_DIR / CLEAN_FILE_TEMPLATE.format(split=split))
        if fieldnames is None:
            fieldnames = current_fieldnames
        split_rows[split] = rows
    assert fieldnames is not None
    return fieldnames, split_rows


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": rounded(min(values)),
        "max": rounded(max(values)),
        "mean": rounded(mean(values)),
        "median": rounded(median(values)),
        "std": rounded(pstdev(values)),
    }


def main() -> None:
    ensure_dirs()
    fieldnames, split_rows = load_cleaned()
    features = numeric_columns(fieldnames)
    all_rows = [row for rows in split_rows.values() for row in rows]

    split_counts = {}
    for split, rows in split_rows.items():
        counts = Counter(row[TARGET_COLUMN] for row in rows)
        split_counts[split] = {cls: counts.get(cls, 0) for cls in CANONICAL_CLASSES}

    figures = []
    figures.append(grouped_bar_chart(FIGURES_DIR / "class_distribution_by_split.svg", "Class Distribution by Split", split_counts))
    figures.append(bar_chart(FIGURES_DIR / "overall_class_distribution.svg", "Overall Class Distribution", dict(Counter(row[TARGET_COLUMN] for row in all_rows))))

    for feature in ("Area", "Perimeter", "Solidity", "Compactness"):
        figures.append(histogram(
            FIGURES_DIR / f"hist_{feature}.svg",
            f"{feature} Distribution After Cleaning",
            [float(row[feature]) for row in all_rows],
        ))

    corr = correlation_matrix(split_rows["train"], features)
    figures.append(heatmap(FIGURES_DIR / "train_feature_correlation.svg", "Training Feature Correlation", corr, features))
    cleaning_report = read_json(REPORTS_DIR / "cleaning_report.json")
    quality_groups = {}
    label_changes = {}
    for split in SPLITS:
        raw_profile = cleaning_report["raw_profile"][split]
        clean_profile = cleaning_report["cleaned_profile"][split]
        quality_groups[split] = {
            "missing": sum(raw_profile["missing_values"].values()),
            "invalid": sum(raw_profile["invalid_numeric_values"].values()),
            "negative": sum(raw_profile["negative_numeric_values"].values()),
        }
        label_changes[split] = {
            "label changes": sum(clean_profile["label_changes"].values()),
            "duplicates removed": clean_profile["duplicates_removed"],
        }
    figures.append(compact_grouped_bar_chart(FIGURES_DIR / "raw_data_quality_issues.svg", "Raw Data Quality Issues", quality_groups))
    figures.append(compact_grouped_bar_chart(FIGURES_DIR / "label_and_duplicate_cleaning.svg", "Label and Duplicate Cleaning", label_changes))

    summary = {
        "rows_by_split": {split: len(rows) for split, rows in split_rows.items()},
        "class_counts_by_split": split_counts,
        "feature_summary_all_splits": {
            feature: describe([float(row[feature]) for row in all_rows])
            for feature in features
        },
        "notable_high_correlations_train_abs_ge_0_95": [],
        "figures": figures,
    }

    high_corr = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            value = corr[a][b]
            if abs(value) >= 0.95:
                high_corr.append({"feature_a": a, "feature_b": b, "correlation": rounded(value)})
    summary["notable_high_correlations_train_abs_ge_0_95"] = sorted(
        high_corr,
        key=lambda item: abs(item["correlation"]),
        reverse=True,
    )

    write_json(REPORTS_DIR / "eda_summary.json", summary)
    print("Generated EDA figures and eda_summary.json")


if __name__ == "__main__":
    main()
