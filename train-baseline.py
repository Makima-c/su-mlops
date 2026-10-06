import json
import pickle
import platform
from pathlib import Path

import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "v0.1"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = ROOT / "models" / MODEL_VERSION


def train_baseline(output_dir: Path = OUTPUT_DIR) -> dict:
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset": "sklearn.datasets.load_diabetes",
        "dataset_scaled": True,
        "feature_names": X.columns.tolist(),
        "target": "progression_index",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "pipeline": {
            "scaler": {
                "class": "StandardScaler",
                "params": model.named_steps["scaler"].get_params(),
            },
            "regressor": {
                "class": "LinearRegression",
                "params": model.named_steps["regressor"].get_params(),
            },
        },
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as f_out:
        pickle.dump(model, f_out)
    for filename, data in [("metrics.json", metrics), ("metadata.json", metadata)]:
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Models saved to {output_dir.resolve()}")
    return metrics


def main():
    train_baseline()


if __name__ == "__main__":
    main()
