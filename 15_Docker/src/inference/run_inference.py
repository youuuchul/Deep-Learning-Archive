from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

TARGET_COLUMN = "Performance Index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with exported model.pkl")
    parser.add_argument("--model-path", default="data/shared/model.pkl")
    parser.add_argument("--test-path", default="data/shared/mission15_test.csv")
    parser.add_argument("--output-path", default="data/shared/result.csv")
    parser.add_argument(
        "--report-path",
        default="data/shared/inference_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    test_path = Path(args.test_path)
    output_path = Path(args.output_path)
    report_path = Path(args.report_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_path.exists():
        fallback_candidates = [
            Path("data/raw/mission15_test.csv"),
            Path("/workspace/data/raw/mission15_test.csv"),
        ]
        existing_fallback = next((p for p in fallback_candidates if p.exists()), None)
        if existing_fallback is not None:
            test_path = existing_fallback
        else:
            raise FileNotFoundError(f"Test data not found: {test_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(test_path)
    feature_df = test_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    prediction = model.predict(feature_df)

    result_df = test_df.copy()
    result_df["Predicted Performance Index"] = np.round(prediction, 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    report = {
        "model_path": str(model_path),
        "test_path": str(test_path),
        "output_path": str(output_path),
        "row_count": int(test_df.shape[0]),
    }

    if TARGET_COLUMN in test_df.columns:
        rmse = root_mean_squared_error(test_df[TARGET_COLUMN], prediction)
        report["rmse_vs_ground_truth"] = float(round(rmse, 6))

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[inference] result saved: {output_path}")
    if "rmse_vs_ground_truth" in report:
        print(f"[inference] RMSE: {report['rmse_vs_ground_truth']:.6f}")


if __name__ == "__main__":
    main()
