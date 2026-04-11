# Stock Price Prediction — Final Definitive Implementation Plan
## AAPL · today (12/04/2026) -> 8 years ago · Three-Paradigm Case Study

---

## Locked Decisions

| Parameter | Value |
|---|---|
| **Ticker** | `AAPL` |
| **Date range** | `2017-01-01` → `2024-12-31` (≈ 2,016 trading days) |
| **Target** | `Target = Close.shift(-1)` — next-day closing price |
| **Primary model** | Linear Regression (OLS) |
| **Comparative model 1** | Support Vector Regression (SVR, Linear kernel) |
| **Comparative model 2** | XGBoost (gblinear booster) |
| **Naive baseline** | `ŷ_{t+1} = Close_t` (DummyRegressor equivalent) — added to NB06 |
| **Train/test split** | Chronological 80/20, no shuffle |
| **Scaling** | StandardScaler for LR and SVR; none needed for gblinear |
| **Feature engineering** | Manual pandas (no `pandas-ta`) |
| **Supplementary check** | Ridge inside NB03, clearly labelled as sensitivity check |
| **Notebooks** | 7 separate `.ipynb` files |

---

## Three-Paradigm Model Design

| Paradigm | Model | Mathematical Foundation | Key Report Argument |
|---|---|---|---|
| **Linear / Parametric** | Linear Regression (OLS) | Minimises sum of squared residuals analytically | Interpretable coefficients; assumes linearity and Gaussian errors |
| **Kernel / Margin** | SVR (Linear kernel) | Fits within ε-insensitive tube, maximizing margin with support vectors | Uses linear kernel to ensure trend extrapolation (prevents RBF drop-off) |
| **Boosting / Linear Base**| XGBoost (gblinear) | Iterative gradient descent with regularized linear base learners | Prevents tree-based flatlining limits (extrapolation failure) by utilizing linear equations |

---

## Why NOT Random Forest or RBF SVR (and why it matters for YOUR case study)

> This section **must appear in NB04/NB05 intros** as academic context-setting.

Random Forest builds its predictions by averaging the targets of training-set leaf nodes. This means it is **structurally incapable of predicting a value higher than the maximum target value seen during training**. Because AAPL's price reaches ~$190 in training but hits ~$220+ in the 2024 test set, an RF model would asymptotically flatline. The same structural failure applies to SVR with an RBF kernel (which drops to zero distance on unseen distributions) and `XGBoost` with `booster='gbtree'`. To fix this, we switch to SVR with a Linear Kernel and XGBoost with `gblinear`.

---

## Directory Structure

Data should be strictly stored locally.

```
stock-price-prediction/
│
├── 00_project_setup_and_data_download.ipynb
├── 01_data_cleaning_and_feature_engineering.ipynb
├── 02_exploratory_data_analysis.ipynb
├── 03_model_linear_regression.ipynb
├── 04_model_svr.ipynb
├── 05_model_xgboost_gblinear.ipynb
├── 06_model_comparison_and_final_evaluation.ipynb
│
├── data/
│   ├── raw_stock_data.csv          ← NB00 output
│   ├── processed_features.csv      ← NB01 output (all intermediates, for EDA)
│   ├── train.csv                   ← NB01 output (12 features + Target)
│   └── test.csv                    ← NB01 output (12 features + Target)
│
└── outputs/
    ├── figures/                    ← all saved charts
    ├── metrics/                    ← CSVs with evaluation metrics
    └── predictions/                ← CSVs with model predictions
```

---

## Shared Local Setup Block (top of every notebook)

```python
# ── LOCAL DEV SETUP ──────────────────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.abspath('.')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
METRICS_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')
PREDS_DIR    = os.path.join(PROJECT_ROOT, 'outputs', 'predictions')

for d in [DATA_DIR, FIGURES_DIR, METRICS_DIR, PREDS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── PROJECT CONSTANTS ─────────────────────────────────────────────────────────
TICKER     = 'AAPL'
START_DATE = '2017-01-01'
END_DATE   = '2024-12-31'
TRAIN_FRAC = 0.80
RANDOM_STATE = 42
```

All figures: `plt.savefig(os.path.join(FIGURES_DIR, 'filename.png'), dpi=150, bbox_inches='tight')`
All metrics: `df.to_csv(os.path.join(METRICS_DIR, 'filename.csv'), index=False)`
All predictions: `df.to_csv(os.path.join(PREDS_DIR, 'filename.csv'), index=False)`

---

## Final Feature Set (12 features — pre-specified, not data-driven)

> Feature selection is **pre-specified by domain knowledge** here and in NB01.
> No feature selection is performed after seeing test data.
> NB02 (EDA) uses training data only for the correlation heatmap.

### Intermediate computed values (used to derive features; kept in `processed_features.csv` for EDA)

| Intermediate | Formula |
|---|---|
| `MA5` | `Close.rolling(5).mean()` |
| `MA20` | `Close.rolling(20).mean()` |
| `Std20` | `Close.rolling(20).std()` |
| `EMA10` | `Close.ewm(span=10, adjust=False).mean()` |
| `EMA20` | `Close.ewm(span=20, adjust=False).mean()` |
| `EMA12` | `Close.ewm(span=12, adjust=False).mean()` |
| `EMA26` | `Close.ewm(span=26, adjust=False).mean()` |
| `MACD_Line` | `EMA12 - EMA26` |
| `MACD_Signal` | `MACD_Line.ewm(span=9, adjust=False).mean()` |
| `BB_Upper` | `MA20 + 2 × Std20` |
| `BB_Lower` | `MA20 - 2 × Std20` |
| `Volume_MA5` | `Volume.rolling(5).mean()` |
| `OBV` | `(sign(Close.diff()) × Volume).fillna(0).cumsum()` |

### Final 12 Model Features

| # | Feature | Formula | Dimension |
|---|---|---|---|
| 1 | `Close` | Raw end-of-day close | Level / autocorrelation anchor |
| 2 | `Daily_Return` | `Close.pct_change()` | Daily percentage return |
| 3 | `Rolling_Std5` | `Daily_Return.rolling(5).std()` | Realized 5-day volatility |
| 4 | `HL_Pct` | `(High - Low) / Close` | Normalized intraday range |
| 5 | `RSI_14` | 14-period RSI (manual) | Momentum oscillator (0–100) |
| 6 | `MACD_Hist` | `MACD_Line - MACD_Signal` | Momentum strength (one MACD variable only) |
| 7 | `BB_Width` | `(BB_Upper - BB_Lower) / MA20` | Bollinger volatility regime |
| 8 | `Vol_Ratio` | `Volume / Volume_MA5` | Volume relative to 5-day average |
| 9 | `OBV_Change` | `OBV.diff()` | Daily net volume direction (stationary) |
| 10 | `Close_MA5_Pct` | `Close / MA5 - 1` | Deviation from short-term trend |
| 11 | `Close_MA20_Pct` | `Close / MA20 - 1` | Deviation from medium-term trend |
| 12 | `EMA_Spread` | `EMA10 - EMA20` | EMA momentum slope |

### Target Variable
`Target = Close.shift(-1)` — next trading day's closing price.
Last row (where Target is NaN) is always dropped.

### Manual RSI formula (pandas, step-by-step)
```python
delta    = df['Close'].diff()
gain     = delta.clip(lower=0)
loss     = (-delta).clip(lower=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs       = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))
```

### Manual MACD formula (pandas, step-by-step)
```python
ema12            = df['Close'].ewm(span=12, adjust=False).mean()
ema26            = df['Close'].ewm(span=26, adjust=False).mean()
macd_line        = ema12 - ema26
macd_signal      = macd_line.ewm(span=9, adjust=False).mean()
df['MACD_Hist']  = macd_line - macd_signal
```

### Manual OBV formula (pandas, step-by-step)
```python
obv              = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
df['OBV_Change'] = obv.diff()
```

### NaN budget
Largest rolling window is MA20 / Std20 (20 rows). After `dropna()`, the dataset loses approximately
**26 rows** (MA20=20, RSI=14, overlap reduces total loss). Final clean dataset: **≈1,990 rows**.

### Chronological 80/20 split
- Training: first 1,592 rows ≈ `2017-02` → `2023-07`
- Testing: last 398 rows ≈ `2023-07` → `2024-12`

---

## Scaling Strategy

| Model | Scaling | Implementation |
|---|---|---|
| **Linear Regression** | `StandardScaler` (fit on X_train only, transform X_test) | Inline, outside Pipeline |
| **Ridge (supplementary)** | Same scaler as LR, reused | Inline |
| **XGBoost (gblinear)** | None — gblinear handles scaling/weights natively | Raw features passed directly |
| **SVR** | `StandardScaler` inside `sklearn.pipeline.Pipeline` | Scaler is fit only on training fold during CV; no leakage |

Brief MinMaxScaler comparison shown in NB03 (1 cell) to illustrate sensitivity of LR coefficient interpretation to scaling choice.

---

## Metrics (identical across all 3 models + baseline)

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | Mean Absolute Error | Average dollar error in price prediction |
| **MSE** | Mean Squared Error | Penalises large errors more heavily |
| **RMSE** | √MSE | Same unit as price (USD); primary ranking metric |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |
| **Directional Accuracy (DA)** | `Mean(Sign(ŷ_{t+1} - Close_t) == Sign(Actual_Close_{t+1} - Close_t))` | Identifies if the model predicted market movement successfully (Crucial for time-series forecasting) |

Naive baseline headline metric: **"Baseline RMSE = X USD, DA = Y%"** so all model metrics are interpreted relative to it.

---

## Notebook-by-Notebook Plan

---

### `00_project_setup_and_data_download.ipynb`

**Purpose:** Reproducible environment setup + raw data acquisition

**Steps:**
1. Install: `!pip install yfinance xgboost scikit-learn`
2. Shared setup block (Paths, constants) using `os` for local environment
3. Import: `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`
4. Download: `yf.download('AAPL', start='2017-01-01', end='2024-12-31', auto_adjust=True)`
5. Keep columns: `Open`, `High`, `Low`, `Close`, `Volume`
   - Note: `auto_adjust=True` means `Close` is already split/dividend-adjusted; `Adj Close` not needed
6. Reset index, ensure `Date` is a datetime column
7. Diagnostics: shape, `.head(5)`, `.tail(5)`, `.info()`, null summary
8. Save: `os.path.join(DATA_DIR, 'raw_stock_data.csv')`

**Output files:**
- `data/raw_stock_data.csv`

---

### `01_data_cleaning_and_feature_engineering.ipynb`

**Purpose:** Data validation + manual computation of all 12 model features

**Steps:**
1. Load `raw_stock_data.csv`, parse dates, sort ascending
2. Inspect for missing values; forward-fill any gaps (trading halts / data artefacts)
3. Confirm no duplicate dates
4. Compute all **intermediate variables** (MA5, ... OBV)
5. Compute all **12 final features** (formulas in table above)
6. Create `Target = df['Close'].shift(-1)`; drop last row
7. Drop all rows with any NaN using `df.dropna()` (≈ first 26 rows)
8. Print: final shape, feature list, null check
9. Save `processed_features.csv` — includes raw OHLCV + all intermediates + all 12 features + Target (for EDA visualization use)
10. Define `FEATURE_COLS` list (12 columns); define `TARGET_COL = 'Target'`
11. Chronological 80/20 split on cleaned data (no shuffle, no stratify)
12. Save `train.csv` and `test.csv` — contain only `Date` + 12 features + `Target`
13. Print: train shape, test shape, train date range, test date range

**Output files:**
- `data/processed_features.csv`
- `data/train.csv`
- `data/test.csv`

---

### `02_exploratory_data_analysis.ipynb`

**Purpose:** Visual pattern discovery + statistical summaries for report

**Data loading:**
- Load `processed_features.csv` (full dataset)
- For correlation heatmap: load `train.csv` only (to avoid leakage into modeling decisions)

**Statistical Cell:**
- `.describe()` on Close, Volume, all 12 features
- Skewness and kurtosis of `Daily_Return`
- Count of rows in train vs test

**Plots (save all to `FIGURES_DIR`):**

| # | Plot | Key Insight Being Shown |
|---|---|---|
| 1 | AAPL closing price (8-year line) | Long-term trend, COVID crash, 2022 correction, recovery |
| 2 | Close + MA5 + MA20 line chart | Short vs medium trend smoothing |
| 3 | EMA10 vs EMA20 + EMA_Spread subplot | EMA crossover as momentum signal |
| 4 | MACD_Line + MACD_Signal + MACD_Hist (3-panel) | Momentum convergence/divergence |
| 5 | RSI_14 with 70/30 horizontal bands | Overbought / oversold regimes |
| 6 | BB_Upper + Close + BB_Lower + BB_Width subplot | Volatility regime changes |
| 7 | Volume bar chart + Volume_MA5 overlay | Volume spikes during key events |
| 8 | Vol_Ratio over time | Abnormal volume days relative to trend |
| 9 | OBV + OBV_Change subplot | Volume accumulation and direction |
| 10 | Daily_Return histogram + KDE with zero line | Return distribution, fat tails, skew |
| 11 | Daily_Return QQ-plot (vs Normal) | Whether normality assumption for LR holds |
| 12 | Returns boxplot by calendar year (8 boxes) | Volatility clustering across years |
| 13 | Correlation heatmap — **training data only** | Feature-target correlations, inter-feature redundancy |
| 14 | Pairplot — top 6 features by abs(corr) with Target | Bivariate scatter relationships |

**Markdown interpretations required under each plot:**
- Price trend narrative (bull/bear periods)
- RSI overbought/oversold events aligning with price peaks/troughs
- MACD crossover relationship to price momentum
- BB_Width spikes during COVID 2020 and Fed rate hike 2022
- Return distribution remarks (non-Gaussian; fat tails; mild negative skew)
- Correlation heatmap: note high corr of `Close` and `Close_MA5_Pct` / `Close_MA20_Pct` with Target; note that this is expected for price-level prediction; note implications for LR multicollinearity

**Output files:**
- `outputs/figures/01_close_price.png` ... `14_pairplot.png`

---

### `03_model_linear_regression.ipynb`

**Purpose:** Primary assignment model — OLS Linear Regression with diagnostics

**Steps:**

**Section A — Data Preparation**
1. Load `train.csv` and `test.csv`
2. Split into `X_train`, `y_train`, `X_test`, `y_test`
3. Brief scaling comparison cell
4. Fit `StandardScaler` on `X_train` only → transform `X_train` and `X_test`

**Section B — Multicollinearity Diagnostics**
5. Compute VIF for all 12 features using `statsmodels.stats.outliers_influence.variance_inflation_factor`
6. Display as sorted table; flag any VIF > 5 (moderate) and VIF > 10 (severe)
7. Markdown: interpret findings — note that `Close` will have massive VIF. This represents "signal swamping" where the autoregressive component eclipses everything else.

**Section C — OLS Linear Regression (Primary)**
8. Train `sklearn.linear_model.LinearRegression`
9. Predict `y_pred_lr` on `X_test` (scaled)
10. Compute: MAE, MSE, RMSE, R², **DA (Directional Accuracy)**
11. Plots:
    - Actual vs Predicted line chart
    - Scatter plot
    - Residual plot
    - Residual histogram + KDE
12. Coefficient table: feature name, coefficient, sorted by absolute magnitude
13. Markdown: Interpret top 3 positive and top 3 negative coefficients. Discuss collinearity limits.

**Section D — Ridge Regression (Supplementary sensitivity check for Multicollinearity)**
14. `RidgeCV(alphas=[0.01, 0.1, 1.0, 10, 100], cv=TimeSeriesSplit(5))`
15. Compare Ridge RMSE, DA vs OLS RMSE, DA
16. Plot Ridge coefficients vs OLS coefficients side-by-side
17. Markdown: explains that Ridge shrinks coefficients dynamically to control collinearity while continuing to extrapolate properly into the 2024 test setup.

**Section E — MinMaxScaler comparison (1 cell)**
18. Refit OLS on MinMaxScaler-scaled data; show coefficient table

**Save:**
- `outputs/predictions/lr_predictions.csv` — Date, y_test, y_pred_lr, Close_t
- `outputs/metrics/lr_metrics.csv` — MAE, MSE, RMSE, R², DA
- All section plots to `outputs/figures/lr_*.png`

---

### `04_model_svr.ipynb`

**Purpose:** Comparative model 1 — Kernel/margin-based regression (SVR, Linear kernel)

**Steps:**

**Section A — Data Preparation**
1. Load `train.csv` and `test.csv`
2. Split `X_train`, `y_train`, `X_test`, `y_test`
3. Markdown: Explain SVR scaling necessity

**Section B — The Fall of RBF (Extrapolation Problem Demonstration)**
4. Demonstrate building an `SVR` pipeline using `kernel='rbf'`.
5. Display predictions via an overlay plot on actual.
6. Markdown: Elaborate carefully that the RBF kernel evaluates distance. Since TS data pushes beyond trained boundaries steadily, distances essentially hit 0 and cause the model to flatline. This is fundamental algorithm design failure for non-stationary prices, motivating the swap to Linear.

**Section C — SVR Pipeline with Linear Kernel**
7. Build `Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='linear'))])`
8. Markdown: Detail how Linear kernel successfully maps hyperplanes indefinitely.

**Section D — Hyperparameter Tuning**
9. Define search space:
   ```python
   svr__C:        [0.1, 1, 10, 50, 100]
   svr__epsilon:  [0.01, 0.05, 0.1, 0.5]
   ```
10. `TimeSeriesSplit(n_splits=5)` + `RandomizedSearchCV`
11. Print best CV params

**Section E — Evaluation**
12. Fit best pipeline
13. Compute metrics including DA.
14. Plot results similar to LR (Actual vs Predicted, Scatter, Residuals).

**Save:**
- `outputs/predictions/svr_predictions.csv`
- `outputs/metrics/svr_metrics.csv`
- All plots to `outputs/figures/svr_*.png`

---

### `05_model_xgboost_gblinear.ipynb`

**Purpose:** Comparative model 2 — Gradient Booster using linear base learners

**Steps:**

**Section A — Data Preparation**
1. Load `train.csv` and `test.csv`

**Section B — The Extrapolation Problem of Trees (Demonstration)**
2. Train a default `XGBRegressor(booster='gbtree')` model.
3. Overlay plot vs Actual.
4. Markdown: Explain that gradient boosted trees hit peak terminal nodes exactly as Random Forests do and cannot shoot higher than max train bounds.

**Section C — XGBoost with Linear Ensembles**
5. Change booster to `gblinear`.
6. Markdown: Expose the paradigm. Using linear additive boosting solves collinearity and scales indefinitely over horizons.

**Section D — Hyperparameter Tuning**
7. Search space for alpha/lambda/learning rate.
8. TimeSeriesSplit + RandomizedSearchCV

**Section E — Evaluation**
9. Extract best bounds. Predict.
10. Metrics including DA.
11. Export metrics and charts.

**Save:**
- `outputs/predictions/xgb_predictions.csv`
- `outputs/metrics/xgb_metrics.csv`
- All plots to `outputs/figures/xgb_*.png`

---

### `06_model_comparison_and_final_evaluation.ipynb`

**Purpose:** Unified comparison across models + baseline; ablation test analysis

**Steps:**

**Section A — Load All Results**
1. Load all 3 metrics CSVs and 3 predictions CSVs. Load targets.

**Section B — Naive Baseline**
2. Construct baseline `y_baseline = X_test['Close']`.
3. Metrics including DA (which should just equate to persistence bounds).

**Section C — Comparison Table**
4. Render stylized global table: Naive Baseline, LR, SVR, XGBoost.

**Section D — Visualizations**
5. Overlay line chart over actual test bounds.
6. RMSE Bar, MAE Bar, R2 Bar, **DA Bar** (Directional accuracy comparative is very critical)
7. KDE Residuals overlay.

**Section E — Ablation Test (The Dominance of Close)**
8. Using Linear Regression, train 3 mini models:
    - Model Alpha: `X = ['Close']` (only the autoregressive anchor).
    - Model Beta: `X = All Features Except Close` (only the technicals).
    - Model Gamma: `X = All 12 Features`.
9. Measure DA & RMSE across all 3.
10. Markdown Insight: Prove that while Close-only yields the best general shape/RMSE, Technical features often capture the swing/DA vastly better. This demonstrates profound academic modeling understanding around signal separation.

**Section F — Analysis Markdown**
11. Tie all insights together logically around non-stationarity, collinearity, and specific bounds limitation of modern ML toolkits.

**Save:**
- `outputs/metrics/final_model_comparison.csv`
- `outputs/metrics/ablation_results.csv`
- All comparison plots to `outputs/figures/comparison_*.png`

---

## Verification Checklist (run after each notebook)

| Notebook | Check |
|---|---|
| NB00 | `raw_stock_data.csv` exists; shape ≈ (2016, 5); zero nulls |
| NB01 | `train.csv` + `test.csv` exist; 12 feature cols + Target col; zero nulls; dates non-overlapping |
| NB02 | All 14 figures saved; heatmap computed from training data only |
| NB03 | `lr_predictions.csv` + `lr_metrics.csv` exist (with DA); VIF & Ridge charts |
| NB04 | `svr_predictions.csv` + `svr_metrics.csv` exist (with DA); RBF experiment documented |
| NB05 | `xgb_predictions.csv` + `xgb_metrics.csv` exist (with DA); gbtree limit exposed |
| NB06 | `final_model_comparison.csv` exists; DA clearly mapped; Ablation tested |
