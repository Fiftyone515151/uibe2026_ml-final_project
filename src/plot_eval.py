from __future__ import annotations

import subprocess

from common import CANONICAL_CLASSES, FIGURES_DIR
from eda import PALETTE, save_figure, svg_escape


def comparison_bar(path, title: str, values: dict[str, float], value_label: str) -> str:
    width, height = 980, 560
    left, right, top, bottom = 92, 36, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    labels = list(values.keys())
    max_value = max(values.values()) if values else 1.0
    if max_value <= 1:
        max_value = 1.0
    step = plot_w / max(len(labels), 1)
    bar_w = step * 0.52
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<text x="22" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 22 {top + plot_h / 2})">{svg_escape(value_label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    for i in range(6):
        value = max_value * i / 5
        y = top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    for idx, label in enumerate(labels):
        value = values[label]
        x = left + idx * step + (step - bar_w) / 2
        h = value / max_value * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{PALETTE[idx % len(PALETTE)]}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="small">{value:.4f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" class="axis">{svg_escape(label)}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def multi_line_chart(
    path,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    x_label: str,
    y_label: str,
) -> str:
    width, height = 1060, 620
    left, right, top, bottom = 94, 190, 72, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_y = min_y - 0.02
    max_y = max_y + 0.02
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        max_y = min_y + 1

    def px(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def py(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" class="axis">{svg_escape(x_label)}</text>')
    parts.append(f'<text x="24" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 24 {top + plot_h / 2})">{svg_escape(y_label)}</text>')

    for i in range(6):
        value = min_y + (max_y - min_y) * i / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#edf1f5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    for i in range(5):
        value = min_x + (max_x - min_x) * i / 4
        x = px(value)
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 26}" text-anchor="middle" class="axis">{value:.2f}</text>')

    for idx, (name, points) in enumerate(series.items()):
        color = PALETTE[idx % len(PALETTE)]
        polyline = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + idx * 26
        parts.append(f'<rect x="{width - right + 36}" y="{legend_y - 12}" width="16" height="16" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{width - right + 60}" y="{legend_y + 1}" class="label">{svg_escape(name)}</text>')

    return save_figure(path, width, height, "\n".join(parts))


def loss_curve(path, title: str, loss_rows: list[dict[str, float]]) -> str:
    width, height = 980, 560
    left, right, top, bottom = 86, 36, 70, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    epochs = [row["epoch"] for row in loss_rows]
    losses = [row["loss"] for row in loss_rows]
    min_loss, max_loss = min(losses), max(losses)
    if min_loss == max_loss:
        max_loss = min_loss + 1
    min_epoch, max_epoch = min(epochs), max(epochs)
    points = []
    for epoch, loss in zip(epochs, losses):
        x = left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_w
        y = top + plot_h - (loss - min_loss) / (max_loss - min_loss) * plot_h
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts = [f'<text x="{left}" y="36" class="title">{svg_escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#a8b3bf"/>')
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#28666e" stroke-width="4"/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#28666e"/>')
    parts.append(f'<text x="{left}" y="{top + plot_h + 32}" class="axis">epoch {min_epoch}</text>')
    parts.append(f'<text x="{width - right}" y="{top + plot_h + 32}" text-anchor="end" class="axis">epoch {max_epoch}</text>')
    parts.append(f'<text x="{left - 10}" y="{top + 4}" text-anchor="end" class="axis">{max_loss:.3f}</text>')
    parts.append(f'<text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" class="axis">{min_loss:.3f}</text>')
    return save_figure(path, width, height, "\n".join(parts))


def confusion_heatmap(path, title: str, matrix: list[list[int]]) -> str:
    cell = 64
    left, top = 150, 80
    width = left + cell * len(CANONICAL_CLASSES) + 42
    height = top + cell * len(CANONICAL_CLASSES) + 170
    max_value = max(max(row) for row in matrix) or 1
    parts = [f'<text x="36" y="36" class="title">{svg_escape(title)}</text>']
    for i, actual in enumerate(CANONICAL_CLASSES):
        y = top + i * cell
        parts.append(f'<text x="{left - 10}" y="{y + 38}" text-anchor="end" class="axis">{actual}</text>')
        parts.append(f'<text x="{left + i * cell + 32}" y="{top + cell * len(CANONICAL_CLASSES) + 20}" text-anchor="end" class="axis" transform="rotate(-45 {left + i * cell + 32} {top + cell * len(CANONICAL_CLASSES) + 20})">{actual}</text>')
        for j, _ in enumerate(CANONICAL_CLASSES):
            value = matrix[i][j]
            intensity = int(245 - 170 * value / max_value)
            color = f"rgb({intensity},{intensity + 8},255)"
            x = left + j * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" class="small">{value}</text>')
    parts.append(f'<text x="{left + cell * 3.5}" y="{height - 24}" text-anchor="middle" class="axis">Predicted label</text>')
    parts.append(f'<text x="24" y="{top + cell * 3.5}" class="axis" transform="rotate(-90 24 {top + cell * 3.5})">Actual label</text>')
    return save_figure(path, width, height, "\n".join(parts))
