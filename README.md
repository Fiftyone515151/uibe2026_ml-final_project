# Dry Bean Classification Final Project

This project is the engineering workspace for the AIT209 final assignment.

## Current Stage

- Raw dirty datasets are stored in `data/raw/`.
- Cleaned datasets are generated into `data/processed/`.
- EDA figures are generated into `outputs/figures/` as PNG files, with SVG source files kept for later editing.
- Data quality and EDA summaries are generated into `outputs/reports/`.

## Run

The current data-cleaning, EDA, and baseline experiment stages use only the Python standard library.

```bash
python3 src/main.py clean
python3 src/main.py eda
python3 src/main.py train
python3 src/main.py robustness
```

Run the complete pipeline:

```bash
python3 src/main.py all
```

Open the project showcase page at `app/index.html`.

## Baseline Algorithms

The baseline experiment includes three multi-class classifiers:

1. Gaussian Naive Bayes
2. Nearest Centroid
3. Softmax Regression

Softmax Regression is the current best baseline on the cleaned test set.

## Robustness Experiment

The robustness experiment retrains every model with corrupted training data and evaluates on the original cleaned test set.

Noise settings:

1. Gaussian feature noise on standardized features.
2. Feature dropout that randomly sets standardized feature values to zero.
3. Label flipping that randomly changes a percentage of training labels.

Noise strengths: `0.00`, `0.05`, `0.10`, `0.20`.

To keep repeated pure-Python training practical, Softmax Regression uses 70 epochs in robustness experiments.

## Cleaning Rules

1. Strip whitespace from column names and class labels.
2. Normalize polluted labels such as `D3RMAS0N`, `S3K3R`, `H0R0Z`, `B0MBAY`, lowercase labels, and labels with trailing spaces.
3. Convert numeric feature columns to floats.
4. Treat non-numeric feature values, missing strings, non-finite values, and negative values as missing.
5. Remove duplicated rows from the training set only.
6. Impute missing feature values with the median calculated from the cleaned training set.
7. Save one cleaned CSV for each split.

## Main Outputs

- `data/processed/dry_bean_clean_train.csv`
- `data/processed/dry_bean_clean_val.csv`
- `data/processed/dry_bean_clean_test.csv`
- `outputs/reports/cleaning_report.json`
- `outputs/reports/eda_summary.json`
- `outputs/reports/baseline_results.json`
- `outputs/reports/robustness_results.json`
- `outputs/metrics/baseline_metrics.csv`
- `outputs/metrics/robustness_metrics.csv`
- `outputs/figures/*.png`
- `app/index.html`
