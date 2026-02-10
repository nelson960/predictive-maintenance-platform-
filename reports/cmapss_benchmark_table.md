# C-MAPSS Benchmark Results (FD001-FD004)

Generated from run artifacts under:

- /predictive_maintaince_platform/models/cmapss/registry`

Model setup (common across runs):

- `rolling_window=5`
- `rul_clip_max=130`
- `failure_horizon_cycles=30`
- `feature_count=123`

## 1) RUL Regression Metrics

Lower is better for `MAE`, `RMSE`, `NASA Score`. Higher is better for `R2`.

| Subset | Run ID | Train Rows | Train Units | Test Units | MAE | RMSE | R2 | NASA Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | `20260210T163643Z` | 20631 | 100 | 100 | 74.92 | 85.25 | -3.3908 | 405647.10 |
| FD002 | `20260210T174653Z` | 53759 | 260 | 259 | 74.76 | 86.87 | -2.8550 | 1528242.34 |
| FD003 | `20260210T174736Z` | 24720 | 100 | 100 | 73.86 | 83.90 | -3.3852 | 435313.32 |
| FD004 | `20260210T174947Z` | 61249 | 249 | 248 | 79.12 | 90.73 | -3.1758 | 1710474.95 |

## 2) Failure-Risk Classification Metrics (`RUL <= 30`)

Higher is better for `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`.

| Subset | Run ID | Accuracy | Precision | Recall | F1 | ROC-AUC | Positive Rate (True) | Positive Rate (Pred) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | `20260210T163643Z` | 0.8700 | 0.6875 | 0.8800 | 0.7719 | 0.9760 | 0.2500 | 0.3200 |
| FD002 | `20260210T174653Z` | 0.3784 | 0.2748 | 1.0000 | 0.4311 | 0.9807 | 0.2355 | 0.8571 |
| FD003 | `20260210T174736Z` | 0.9100 | 0.7200 | 0.9000 | 0.8000 | 0.9819 | 0.2000 | 0.2500 |
| FD004 | `20260210T174947Z` | 0.4113 | 0.2663 | 1.0000 | 0.4206 | 0.9754 | 0.2137 | 0.8024 |

## 3) Quick Interpretation

- RUL regression is currently weak across all subsets (high error, strongly negative `R2`).
- Classification ROC-AUC is consistently high (~0.975-0.982), so ranking near-failure vs non-failure is strong.
- FD002 and FD004 are over-triggering positives (`positive_rate_pred` >> `positive_rate_true`), driving high recall but low precision/accuracy.

## 4) Source Files

Each row is sourced from:

- `models/cmapss/registry/<run_id>/metadata.json`
- `models/cmapss/registry/<run_id>/metrics.json`
