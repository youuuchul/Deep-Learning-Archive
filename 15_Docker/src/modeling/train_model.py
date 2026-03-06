from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "Performance Index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regression model and export model/data artifacts."
    )
    parser.add_argument("--train-path", default="/workspace/data/raw/mission15_train.csv")
    parser.add_argument("--test-path", default="/workspace/data/raw/mission15_test.csv")
    parser.add_argument("--shared-dir", default="/workspace/data/shared")
    parser.add_argument("--model-name", default="model.pkl")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        TARGET_COLUMN,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_pipeline(x_df: pd.DataFrame) -> Pipeline:
    categorical_features = (
        x_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )
    numeric_features = [c for c in x_df.columns if c not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", LinearRegression()),
        ]
    )


def build_eda_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "target_stats": df[TARGET_COLUMN].describe().round(4).to_dict(),
        "numeric_stats": numeric_df.describe().round(4).to_dict(),
    }
    return summary


def write_eda_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# EDA Summary",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Columns: {summary['column_count']}",
        "",
        "## Missing Values",
    ]

    missing_values = summary["missing_values"]
    for column_name, missing_count in missing_values.items():
        lines.append(f"- {column_name}: {missing_count}")

    lines.extend(["", "## Target Stats (Performance Index)"])
    for key, value in summary["target_stats"].items():
        lines.append(f"- {key}: {value}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    shared_dir = Path(args.shared_dir)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Put mission15_train.csv in data/raw/."
        )

    df = pd.read_csv(train_path)
    validate_columns(df)

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)

    valid_pred = pipeline.predict(x_valid)
    rmse = root_mean_squared_error(y_valid, valid_pred)

    shared_dir.mkdir(parents=True, exist_ok=True)

    model_path = shared_dir / args.model_name
    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)

    metrics = {
        "rmse": float(round(rmse, 6)),
        "train_rows": int(x_train.shape[0]),
        "valid_rows": int(x_valid.shape[0]),
        "feature_columns": x.columns.tolist(),
    }
    (shared_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    eda_summary = build_eda_summary(df)
    (shared_dir / "eda_summary.json").write_text(
        json.dumps(eda_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_eda_markdown(eda_summary, shared_dir / "eda_summary.md")

    if test_path.exists():
        copied_test_path = shared_dir / test_path.name
        shutil.copy2(test_path, copied_test_path)

    print(f"[modeling] model saved: {model_path}")
    print(f"[modeling] rmse: {rmse:.6f}")
    print(f"[modeling] metrics saved: {shared_dir / 'metrics.json'}")
    if test_path.exists():
        print(f"[modeling] {test_path.name} copied to: {copied_test_path}")


if __name__ == "__main__":
    main()
