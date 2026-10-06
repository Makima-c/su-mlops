import json
import math
import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL_DIR = ROOT / "models" / "v0.2"
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
with (MODEL_DIR / "model.pkl").open("rb") as source:
    model = pickle.load(source)

FEATURE_NAMES = metadata["feature_names"]
MODEL_VERSION = metadata["model_version"]
if list(model.feature_names_in_) != FEATURE_NAMES:
    raise ValueError("Model and metadata feature order do not match")

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=MODEL_VERSION)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(
            error="Invalid input", details={"body": "Expected a JSON object"}
        ), 422

    errors = {}
    values = []
    for name in FEATURE_NAMES:
        if name not in payload:
            errors[name] = "Required field is missing"
            continue
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors[name] = "Expected a finite number"
            continue
        try:
            value = float(value)
        except (OverflowError, ValueError):
            errors[name] = "Expected a finite number"
            continue
        if not math.isfinite(value):
            errors[name] = "Expected a finite number"
            continue
        values.append(value)

    if errors:
        return jsonify(error="Invalid input", details=errors), 422

    features = pd.DataFrame([values], columns=FEATURE_NAMES)
    try:
        prediction = float(model.predict(features)[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed"), 500
    return jsonify(prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
