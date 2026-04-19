from pathlib import Path
import json
from typing import Any
import pandas as pd
import numpy as np


RESULTS_ROOT = Path("data/results")
if not RESULTS_ROOT.exists():
    raise FileNotFoundError(f"Could not find results directory: {RESULTS_ROOT.resolve()}")

METRICS = ["MAE", "RMSE", "MAPE", "R2"]
ERROR_METRICS = ["MAE", "RMSE", "MAPE"]



def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient="records")
        return [_to_jsonable(r) for r in records]

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_result_json(
    result: Any,
    *,
    default_filename: str = "model_result.json",
    indent: int = 2,
) -> Path:
    folder_name = default_filename.split("_")[0]
    path = Path("data/results") / folder_name / default_filename
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(result)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")



def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_model_name(raw_dir: str) -> str:
    mapping = {
        "elasticnet": "ElasticNet",
        "Linear Regression": "Linear Regression",
        "Random Forest": "Random Forest",
        "SVR-linear": "SVR-linear",
    }
    return mapping.get(raw_dir, raw_dir)


def _extract_stage(filename: str) -> str:
    low = filename.lower()
    if "train" in low:
        return "train"
    if "test" in low:
        return "test"
    return "test"


def _extract_variant(filename: str) -> str:
    low = filename.lower()
    if "baseline" in low:
        return "baseline"
    if "tuned" in low:
        return "tuned"
    # Files without explicit variant (for some models) are treated as baseline
    return "baseline"


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]):
        model_name = _canonical_model_name(model_dir.name)
        for file in model_dir.glob("*_metrics.json"):
            payload = _load_json(file)
            if not payload:
                continue
            rec = payload[0]
            row = {
                "model": model_name,
                "variant": _extract_variant(file.name),
                "stage": _extract_stage(file.name),
            }
            for m in METRICS:
                row[m] = float(rec.get(m, np.nan))
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No metric JSON files found under data/results")

    # Keep best available row per model/variant/stage (latest alphabetical if duplicates)
    df = df.sort_values(["model", "variant", "stage"]).drop_duplicates(
        subset=["model", "variant", "stage"], keep="last"
    )
    return df.reset_index(drop=True)


def load_boxplot_stats() -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]):
        model_name = _canonical_model_name(model_dir.name)
        files = sorted(model_dir.glob("*boxplot_stats.json"))
        if not files:
            continue
        payload = _load_json(files[-1])
        if not payload:
            continue
        rec = payload[0]
        rows.append({
            "model": model_name,
            "q1": float(rec["Q1 (25%)"]),
            "median": float(rec["Median (50%)"]),
            "q3": float(rec["Q3 (75%)"]),
            "whislo": float(rec["Lower whisker (min within fence)"]),
            "whishi": float(rec["Upper whisker (max within fence)"]),
        })
    return pd.DataFrame(rows)


def load_feature_importance() -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]):
        model_name = _canonical_model_name(model_dir.name)
        files = sorted(model_dir.glob("*feature_importance.json"))
        if not files:
            continue

        payload = _load_json(files[-1])

        for rec in payload:
            val = (
                rec.get("coef_scaled")
                if rec.get("coef_scaled") is not None else
                rec.get("coef_original_units")
                if rec.get("coef_original_units") is not None else
                rec.get("importance")
            )

            if val is None:
                continue

            rows.append({
                "model": model_name,
                "feature": rec.get("feature", "unknown"),
                "importance": float(abs(val)),
            })

    return pd.DataFrame(rows)