import json
import pickle
import platform
from pathlib import Path

import sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "v0.2"
RANDOM_STATE = 42
TEST_SIZE = 0.2
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 5],
}
OUTPUT_DIR = ROOT / "models" / MODEL_VERSION


def train_improved(output_dir: Path = OUTPUT_DIR) -> dict:
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Tune on training folds; keep the test set held out.
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    rmse = float(root_mean_squared_error(y_test, model.predict(X_test)))
    metrics = {
        "rmse": rmse,
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
        "model": {
            "class": "RandomForestRegressor",
            "params": model.get_params(),
        },
        "tuning": {
            "param_grid": PARAM_GRID,
            "cv_folds": 5,
            "cv_shuffle": True,
            "cv_random_state": RANDOM_STATE,
            "cv_rmse": float(-search.best_score_),
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

    print(f"RMSE: {rmse:.4f}")
    print(f"Selected parameters: {search.best_params_}")
    print(f"Models saved to {output_dir.resolve()}")
    return metrics


if __name__ == "__main__":
    train_improved()
