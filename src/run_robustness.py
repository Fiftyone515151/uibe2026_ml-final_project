from __future__ import annotations

import copy
import random
import time

from common import (
    CANONICAL_CLASSES,
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    write_json,
)
from ml_core import (
    MODEL_REGISTRY,
    SoftmaxRegression,
    StandardScaler,
    accuracy,
    load_split,
    macro_f1,
    measure_prediction,
    write_metrics_csv,
)
from plot_eval import multi_line_chart
from run_experiments import evaluate_model


NOISE_TYPES = {
    "gaussian_feature_noise": "Gaussian feature noise on standardized training features",
    "feature_dropout": "Randomly set standardized feature values to zero",
    "label_flip": "Randomly flip a percentage of training labels",
}
STRENGTHS = [0.0, 0.05, 0.10, 0.20]
SEED = 20260618


def corrupt_features_gaussian(x: list[list[float]], strength: float, rng: random.Random) -> list[list[float]]:
    if strength == 0:
        return copy.deepcopy(x)
    return [[value + rng.gauss(0.0, strength) for value in row] for row in x]


def corrupt_features_dropout(x: list[list[float]], strength: float, rng: random.Random) -> list[list[float]]:
    if strength == 0:
        return copy.deepcopy(x)
    return [[0.0 if rng.random() < strength else value for value in row] for row in x]


def corrupt_labels(y: list[str], strength: float, rng: random.Random) -> list[str]:
    if strength == 0:
        return y[:]
    out = []
    for label in y:
        if rng.random() < strength:
            choices = [cls for cls in CANONICAL_CLASSES if cls != label]
            out.append(rng.choice(choices))
        else:
            out.append(label)
    return out


def make_model(model_name: str):
    if model_name == SoftmaxRegression.name:
        return SoftmaxRegression(epochs=70, learning_rate=0.08, l2=0.0005, seed=42)
    return MODEL_REGISTRY[model_name]()


def main() -> None:
    ensure_dirs()
    x_train, y_train, features = load_split("train")
    x_test, y_test, _ = load_split("test")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rows = []
    detailed = {
        "description": "Models are retrained on corrupted training data and evaluated on the original cleaned test set.",
        "noise_types": NOISE_TYPES,
        "strengths": STRENGTHS,
        "softmax_epochs_for_robustness": 70,
        "features": features,
        "results": [],
    }

    for noise_type in NOISE_TYPES:
        for strength in STRENGTHS:
            for model_name in MODEL_REGISTRY:
                rng = random.Random(f"{SEED}-{noise_type}-{strength}-{model_name}")
                noisy_x = copy.deepcopy(x_train_scaled)
                noisy_y = y_train[:]
                if noise_type == "gaussian_feature_noise":
                    noisy_x = corrupt_features_gaussian(x_train_scaled, strength, rng)
                elif noise_type == "feature_dropout":
                    noisy_x = corrupt_features_dropout(x_train_scaled, strength, rng)
                elif noise_type == "label_flip":
                    noisy_y = corrupt_labels(y_train, strength, rng)

                model = make_model(model_name)
                print(f"Training {model_name} | {noise_type} | strength={strength:.2f}")
                start = time.perf_counter()
                history = model.fit(noisy_x, noisy_y)
                train_seconds = time.perf_counter() - start

                train_pred, _, _ = measure_prediction(model, x_train_scaled)
                test_eval = evaluate_model(model, x_test_scaled, y_test)
                train_accuracy = accuracy(y_train, train_pred)

                row = {
                    "noise_type": noise_type,
                    "strength": f"{strength:.2f}",
                    "model": model_name,
                    "train_accuracy_on_clean_train": f"{train_accuracy:.6f}",
                    "test_accuracy": f"{test_eval['accuracy']:.6f}",
                    "test_macro_f1": f"{test_eval['macro_f1']:.6f}",
                    "train_test_accuracy_gap": f"{train_accuracy - test_eval['accuracy']:.6f}",
                    "train_seconds": f"{train_seconds:.6f}",
                    "test_prediction_seconds": f"{test_eval['prediction_time_seconds']:.6f}",
                    "test_per_sample_ms": f"{test_eval['per_sample_ms']:.6f}",
                }
                rows.append(row)
                detailed["results"].append(
                    {
                        **row,
                        "history": history,
                        "test_confusion_matrix": test_eval["confusion_matrix"],
                    }
                )

    clean_baselines = {
        row["model"]: float(row["test_accuracy"])
        for row in rows
        if row["noise_type"] == "gaussian_feature_noise" and row["strength"] == "0.00"
    }
    for row in rows:
        baseline = clean_baselines[row["model"]]
        row["accuracy_drop_vs_clean"] = f"{baseline - float(row['test_accuracy']):.6f}"
    for item in detailed["results"]:
        baseline = clean_baselines[item["model"]]
        item["accuracy_drop_vs_clean"] = f"{baseline - float(item['test_accuracy']):.6f}"

    write_metrics_csv(METRICS_DIR / "robustness_metrics.csv", rows)
    write_json(REPORTS_DIR / "robustness_results.json", detailed)

    for noise_type in NOISE_TYPES:
        series = {}
        drop_series = {}
        for model_name in MODEL_REGISTRY:
            points = []
            drop_points = []
            for strength in STRENGTHS:
                match = next(
                    row for row in rows
                    if row["noise_type"] == noise_type
                    and row["model"] == model_name
                    and row["strength"] == f"{strength:.2f}"
                )
                points.append((strength, float(match["test_accuracy"])))
                drop_points.append((strength, float(match["accuracy_drop_vs_clean"])))
            series[model_name] = points
            drop_series[model_name] = drop_points
        multi_line_chart(
            FIGURES_DIR / f"robustness_accuracy_{noise_type}.svg",
            f"Robustness Accuracy - {noise_type}",
            series,
            "Noise strength",
            "Test accuracy",
        )
        multi_line_chart(
            FIGURES_DIR / f"robustness_drop_{noise_type}.svg",
            f"Accuracy Drop - {noise_type}",
            drop_series,
            "Noise strength",
            "Accuracy drop",
        )

    print(f"Wrote {len(rows)} robustness metric rows.")


if __name__ == "__main__":
    main()
