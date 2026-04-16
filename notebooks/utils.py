from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def load_data():
    CONFIG_PATH = Path("../data/processed/modeling_config.json")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    data_csv = Path(cfg["data_csv"])
    df = pd.read_csv(data_csv, index_col=0, parse_dates=True)

    feature_cols = cfg["feature_cols"]
    target_col = cfg["target_col"]

    return df, feature_cols, target_col


def train_test_split(df, feature_cols, target_col, test_size=0.2):
    n_samples = len(df)
    n_test = int(n_samples * test_size)

    df_train = df.iloc[:-n_test]
    df_test = df.iloc[-n_test:]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    print("Configured target:", target_col)
    print("Configured features:", feature_cols)

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows:  {len(X_test)}")

    print(f"Train period: {df_train.index.min().date()} -> {df_train.index.max().date()}")
    print(f"Test period:  {df_test.index.min().date()} -> {df_test.index.max().date()}")

    return X_train, y_train, X_test, y_test


def regression_metrics(y_true, y_pred, model_name: str) -> pd.DataFrame:
    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    return pd.DataFrame([metrics]).round(4)


def visualize_boxplot_stats(x):
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    lower_whisker = np.min(x[x >= lower_fence])
    upper_whisker = np.max(x[x <= upper_fence])

    stats_df = pd.DataFrame(
        {
            "value": [q1, q2, q3, iqr, lower_fence, upper_fence, lower_whisker, upper_whisker]
        },
        index=[
            "Q1 (25%)",
            "Median (50%)",
            "Q3 (75%)",
            "IQR (Q3-Q1)",
            "Lower fence (Q1-1.5*IQR)",
            "Upper fence (Q3+1.5*IQR)",
            "Lower whisker (min within fence)",
            "Upper whisker (max within fence)",
        ],
    )

    return stats_df.round(4)


def visualize_predictions(y, y_pred, model_name: str):
    y_test_arr = np.asarray(y)
    best_pred_arr = np.asarray(y_pred)
    pct_diff = np.where(y_test_arr != 0, ((best_pred_arr - y_test_arr) / y_test_arr) * 100, np.nan)

    valid_pct_diff = pct_diff[~np.isnan(pct_diff)]
    if valid_pct_diff.size == 0:
        valid_pct_diff = np.array([0.0])
    abs_limit = np.max(np.abs(valid_pct_diff))
    if abs_limit == 0:
        abs_limit = 1

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    sns.histplot(valid_pct_diff, bins=40, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_title("Percentage Difference Distribution ({model_name})".format(model_name=model_name))
    axes[0].set_ylabel("Count")

    sns.boxplot(x=valid_pct_diff, ax=axes[1], color="#1f77b4")
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("% Difference (Prediction - Actual) / Actual")

    axes[0].set_xlim(-abs_limit, abs_limit)
    axes[1].set_xlim(-abs_limit, abs_limit)

    plt.tight_layout()
    plt.show()

    visualize_boxplot_stats(valid_pct_diff)






def visualize_feature_importance(model, scalar, feature_cols, model_name: str):
    scaler = scalar
    estimator = model

    coef_scaled = np.ravel(estimator.coef_)

    # If a scaler exists, convert coefficients back to original feature units
    if scaler is not None and hasattr(scaler, "scale_"):
        coef_original_units = coef_scaled / scaler.scale_
    else:
        coef_original_units = coef_scaled

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "coef_scaled": coef_scaled,
                "coef_original_units": coef_original_units,
            }
        )
        .sort_values("coef_original_units", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df.head(20), x="coef_original_units", y="feature", color="#1f77b4")
    plt.title("{model_name} Coefficients (approx. original units) — Top 20".format(model_name=model_name))
    plt.tight_layout()
    plt.show()

    return coef_df



def train_test_model(model, X_train, y_train, X_test, y_test):
    model_name = type(model).__name__
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics_test = regression_metrics(y_test, pred_test, f"{model_name} (test)")
    metrics_train = regression_metrics(y_train, pred_train, f"{model_name} (train)")

    print(metrics_train.round(4))
    print(metrics_test.round(4))

    return pred_train, pred_test, model



def visualize_prediction_vs_actual(y_test, y_pred, target_col_name, model_name: str):
    pred_df = pd.DataFrame(index=y_test.index)
    pred_df["Actual"] = y_test
    pred_df[model_name] = y_pred    

    plt.figure(figsize=(14, 6))
    plt.plot(pred_df.index, pred_df["Actual"], label="Actual", linewidth=2, color="#7E7A7A")
    plt.plot(pred_df.index, pred_df[model_name], label=model_name, color="#00FF80")
    plt.title(f"Test Period: Actual vs Model Predictions (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel(target_col_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


def time_series_grid_search(
    estimator,
    param_grid,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    *,
    scoring: str = "neg_root_mean_squared_error",
    n_splits: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """
    Run an exhaustive GridSearchCV using TimeSeriesSplit (forward-in-time CV).

    Returns a dict with the fitted GridSearchCV, best estimator/params, best CV score,
    and (optionally) test predictions + metrics.
    """
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=n_jobs,
        verbose=verbose,
    )

    grid.fit(X_train, y_train)

    best_cv_score = grid.best_score_
    is_negated_metric = isinstance(scoring, str) and scoring.startswith("neg_")
    best_cv_value = -best_cv_score if is_negated_metric else best_cv_score
    metric_name = scoring.replace("neg_", "") if is_negated_metric else scoring

    print(f"Best CV {metric_name}:", best_cv_value)
    print("Best params:", grid.best_params_)

    best_estimator = grid.best_estimator_

    y_pred_test = None
    metrics_test = None
    label = f"{type(best_estimator).__name__} (tuned via TimeSeriesSplit)"
    if X_test is not None:
        y_pred_test = best_estimator.predict(X_test)
        metrics_test = regression_metrics(y_test, y_pred_test, label)
        print(metrics_test)

    return best_estimator, y_pred_test
