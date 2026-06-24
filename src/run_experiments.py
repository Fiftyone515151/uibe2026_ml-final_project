from __future__ import annotations

import json
import time

from common import FIGURES_DIR, METRICS_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs, rounded, write_json
from ml_core import (
    MODEL_REGISTRY,
    StandardScaler,
    accuracy,
    confusion_matrix,
    load_split,
    macro_f1,
    measure_prediction,
    save_model,
    write_metrics_csv,
)
from plot_eval import comparison_bar, confusion_heatmap, loss_curve


def evaluate_model(model, x, y) -> dict[str, object]:
    pred, elapsed, per_sample_ms = measure_prediction(model, x)
    return {
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "prediction_time_seconds": elapsed,
        "per_sample_ms": per_sample_ms,
        "confusion_matrix": confusion_matrix(y, pred),
        "predictions": pred,
    }


def main() -> None:
    ensure_dirs()
    x_train, y_train, features = load_split("train")
    x_val, y_val, _ = load_split("val")
    x_test, y_test, _ = load_split("test")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    metrics_rows = []
    detailed = {
        "features": features,
        "models": {},
    }

    for model_name, model_cls in MODEL_REGISTRY.items():
        print(f"Training {model_name}...")
        model = model_cls()
        start = time.perf_counter()
        history = model.fit(x_train_scaled, y_train)
        train_seconds = time.perf_counter() - start

        train_eval = evaluate_model(model, x_train_scaled, y_train)
        val_eval = evaluate_model(model, x_val_scaled, y_val)
        test_eval = evaluate_model(model, x_test_scaled, y_test)

        metrics_rows.append(
            {
                "model": model_name,
                "train_accuracy": f"{train_eval['accuracy']:.6f}",
                "val_accuracy": f"{val_eval['accuracy']:.6f}",
                "test_accuracy": f"{test_eval['accuracy']:.6f}",
                "test_macro_f1": f"{test_eval['macro_f1']:.6f}",
                "train_test_accuracy_gap": f"{train_eval['accuracy'] - test_eval['accuracy']:.6f}",
                "train_seconds": f"{train_seconds:.6f}",
                "test_prediction_seconds": f"{test_eval['prediction_time_seconds']:.6f}",
                "test_per_sample_ms": f"{test_eval['per_sample_ms']:.6f}",
            }
        )

        detailed["models"][model_name] = {
            "display_name": model.display_name,
            "has_loss_curve": model.has_loss_curve,
            "history": history,
            "train": {k: v for k, v in train_eval.items() if k != "predictions"},
            "val": {k: v for k, v in val_eval.items() if k != "predictions"},
            "test": {k: v for k, v in test_eval.items() if k != "predictions"},
        }
        save_model(model_name, {"model": model, "scaler": scaler, "features": features})

        confusion_heatmap(
            FIGURES_DIR / f"confusion_{model_name}.svg",
            f"Confusion Matrix - {model.display_name}",
            test_eval["confusion_matrix"],
        )
        if history["loss"]:
            loss_curve(
                FIGURES_DIR / f"loss_{model_name}.svg",
                f"Training Loss - {model.display_name}",
                history["loss"],
            )

    write_metrics_csv(METRICS_DIR / "baseline_metrics.csv", metrics_rows)
    write_json(REPORTS_DIR / "baseline_results.json", detailed)

    test_accuracy = {row["model"]: float(row["test_accuracy"]) for row in metrics_rows}
    pred_speed = {row["model"]: float(row["test_per_sample_ms"]) for row in metrics_rows}
    comparison_bar(FIGURES_DIR / "baseline_test_accuracy.svg", "Baseline Test Accuracy", test_accuracy, "Accuracy")
    comparison_bar(FIGURES_DIR / "baseline_inference_speed_ms.svg", "Baseline Inference Speed", pred_speed, "Milliseconds / sample")

    print(json.dumps(metrics_rows, indent=2))


if __name__ == "__main__":
    main()
