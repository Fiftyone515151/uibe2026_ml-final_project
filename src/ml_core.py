from __future__ import annotations

import csv
import math
import pickle
import random
import time
from collections import Counter
from pathlib import Path

from common import (
    CANONICAL_CLASSES,
    CLEAN_FILE_TEMPLATE,
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET_COLUMN,
    ensure_dirs,
    numeric_columns,
    read_csv,
)


def load_split(split: str) -> tuple[list[list[float]], list[str], list[str]]:
    fieldnames, rows = read_csv(PROCESSED_DIR / CLEAN_FILE_TEMPLATE.format(split=split))
    features = numeric_columns(fieldnames)
    x = [[float(row[col]) for col in features] for row in rows]
    y = [row[TARGET_COLUMN] for row in rows]
    return x, y, features


class StandardScaler:
    def __init__(self) -> None:
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, x: list[list[float]]) -> None:
        n_features = len(x[0])
        self.means = []
        self.stds = []
        for j in range(n_features):
            values = [row[j] for row in x]
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(var) or 1.0
            self.means.append(mean)
            self.stds.append(std)

    def transform(self, x: list[list[float]]) -> list[list[float]]:
        return [
            [(value - self.means[j]) / self.stds[j] for j, value in enumerate(row)]
            for row in x
        ]

    def fit_transform(self, x: list[list[float]]) -> list[list[float]]:
        self.fit(x)
        return self.transform(x)


class GaussianNaiveBayes:
    name = "gaussian_nb"
    display_name = "Gaussian Naive Bayes"
    has_loss_curve = False

    def fit(self, x: list[list[float]], y: list[str]) -> dict[str, object]:
        n = len(x)
        n_features = len(x[0])
        self.classes = CANONICAL_CLASSES[:]
        self.priors = {}
        self.means = {}
        self.vars = {}
        for cls in self.classes:
            rows = [row for row, label in zip(x, y) if label == cls]
            self.priors[cls] = len(rows) / n
            self.means[cls] = []
            self.vars[cls] = []
            for j in range(n_features):
                values = [row[j] for row in rows]
                mean = sum(values) / len(values)
                var = sum((value - mean) ** 2 for value in values) / len(values)
                self.means[cls].append(mean)
                self.vars[cls].append(max(var, 1e-9))
        return {"loss": []}

    def predict_one(self, row: list[float]) -> str:
        best_cls = None
        best_score = -float("inf")
        for cls in self.classes:
            score = math.log(self.priors[cls] or 1e-12)
            for j, value in enumerate(row):
                var = self.vars[cls][j]
                mean = self.means[cls][j]
                score += -0.5 * math.log(2 * math.pi * var) - ((value - mean) ** 2) / (2 * var)
            if score > best_score:
                best_score = score
                best_cls = cls
        return str(best_cls)

    def predict(self, x: list[list[float]]) -> list[str]:
        return [self.predict_one(row) for row in x]


class NearestCentroid:
    name = "nearest_centroid"
    display_name = "Nearest Centroid"
    has_loss_curve = False

    def fit(self, x: list[list[float]], y: list[str]) -> dict[str, object]:
        self.classes = CANONICAL_CLASSES[:]
        n_features = len(x[0])
        self.centroids = {}
        for cls in self.classes:
            rows = [row for row, label in zip(x, y) if label == cls]
            self.centroids[cls] = [
                sum(row[j] for row in rows) / len(rows)
                for j in range(n_features)
            ]
        return {"loss": []}

    def predict_one(self, row: list[float]) -> str:
        best_cls = None
        best_dist = float("inf")
        for cls, centroid in self.centroids.items():
            dist = sum((value - centroid[j]) ** 2 for j, value in enumerate(row))
            if dist < best_dist:
                best_dist = dist
                best_cls = cls
        return str(best_cls)

    def predict(self, x: list[list[float]]) -> list[str]:
        return [self.predict_one(row) for row in x]


class SoftmaxRegression:
    name = "softmax_regression"
    display_name = "Softmax Regression"
    has_loss_curve = True

    def __init__(self, epochs: int = 120, learning_rate: float = 0.08, l2: float = 0.0005, seed: int = 42) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.classes = CANONICAL_CLASSES[:]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def fit(self, x: list[list[float]], y: list[str]) -> dict[str, object]:
        rng = random.Random(self.seed)
        n_features = len(x[0])
        n_classes = len(self.classes)
        self.weights = [[rng.uniform(-0.01, 0.01) for _ in range(n_features)] for _ in range(n_classes)]
        self.bias = [0.0 for _ in range(n_classes)]
        losses = []
        indices = list(range(len(x)))

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            total_loss = 0.0
            correct = 0
            for idx in indices:
                row = x[idx]
                target = self.class_to_idx[y[idx]]
                probs = self._predict_proba_one(row)
                total_loss += -math.log(max(probs[target], 1e-12))
                pred = max(range(n_classes), key=lambda k: probs[k])
                if pred == target:
                    correct += 1
                for k in range(n_classes):
                    error = probs[k] - (1.0 if k == target else 0.0)
                    for j, value in enumerate(row):
                        grad = error * value + self.l2 * self.weights[k][j]
                        self.weights[k][j] -= self.learning_rate * grad
                    self.bias[k] -= self.learning_rate * error
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                losses.append(
                    {
                        "epoch": epoch + 1,
                        "loss": total_loss / len(x),
                        "train_accuracy": correct / len(x),
                    }
                )
        return {"loss": losses}

    def _predict_proba_one(self, row: list[float]) -> list[float]:
        logits = []
        for weights, bias in zip(self.weights, self.bias):
            logits.append(sum(w * value for w, value in zip(weights, row)) + bias)
        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def predict_one(self, row: list[float]) -> str:
        probs = self._predict_proba_one(row)
        return self.classes[max(range(len(probs)), key=lambda idx: probs[idx])]

    def predict(self, x: list[list[float]]) -> list[str]:
        return [self.predict_one(row) for row in x]


MODEL_REGISTRY = {
    GaussianNaiveBayes.name: GaussianNaiveBayes,
    NearestCentroid.name: NearestCentroid,
    SoftmaxRegression.name: SoftmaxRegression,
}


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    index = {cls: i for i, cls in enumerate(CANONICAL_CLASSES)}
    matrix = [[0 for _ in CANONICAL_CLASSES] for _ in CANONICAL_CLASSES]
    for actual, pred in zip(y_true, y_pred):
        matrix[index[actual]][index[pred]] += 1
    return matrix


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = []
    for cls in CANONICAL_CLASSES:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == cls and p == cls)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != cls and p == cls)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def measure_prediction(model, x: list[list[float]]) -> tuple[list[str], float, float]:
    start = time.perf_counter()
    pred = model.predict(x)
    elapsed = time.perf_counter() - start
    return pred, elapsed, elapsed / len(x) * 1000


def save_model(name: str, payload: object) -> None:
    ensure_dirs()
    with (MODELS_DIR / f"{name}.pkl").open("wb") as f:
        pickle.dump(payload, f)


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
