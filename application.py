import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ridge_regressor_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts() -> Tuple[object, object]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    return model, scaler


def get_feature_names(scaler) -> List[str]:
    names: Optional[np.ndarray] = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in list(names)]

    n_features = None
    for attr in ("n_features_in_", "n_features"):
        if hasattr(scaler, attr):
            n_features = int(getattr(scaler, attr))
            break
    if not n_features:
        n_features = 10
    return [f"feature_{i+1}" for i in range(n_features)]


app = Flask(__name__, template_folder=os.path.join(APP_ROOT, "templates"))

try:
    model, scaler = load_artifacts()
    FEATURE_NAMES = get_feature_names(scaler)
except Exception as exc:
    model = None
    scaler = None
    FEATURE_NAMES = []
    _artifact_error = str(exc)
else:
    _artifact_error = None

# Raw dataset feature names (exclude target 'FWI')
# These are the raw inputs collected from the UI; server will map to the
# model/scaler expected order using FEATURE_NAMES.
RAW_FEATURE_NAMES = [
    "day",
    "month",
    "year",
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "Classes",
    "region",
]

# Only display fields that the trained scaler/model actually expects
DISPLAY_RAW_FEATURE_NAMES = [name for name in RAW_FEATURE_NAMES if name in FEATURE_NAMES]

# Suggested defaults shown as placeholders in the UI
SUGGESTED_DEFAULTS = {
    "day": 1,
    "month": 6,
    "year": 2012,
    "Temperature": 25,
    "RH": 60,
    "Ws": 10,
    "Rain": 0,
    "FFMC": 65.0,
    "DMC": 4.0,
    "ISI": 1.0,
    "Classes": 0,
    "region": 0,
}


def _coerce_raw_value(name: str, value: str) -> float:
    """Coerce incoming raw form/api values to numeric floats.

    Special handling:
    - Classes: accepts 0/1 or strings 'not fire'/'fire'
    - region: coerced to 0/1
    """
    if value is None or value == "":
        raise ValueError(f"Missing value for {name}")

    if name == "Classes":
        v = str(value).strip().lower()
        if v in {"0", "0.0", "not fire", "nofire", "no_fire"}:
            return 0.0
        if v in {"1", "1.0", "fire"}:
            return 1.0
        return float(value)

    if name == "region":
        v = float(value)
        return 1.0 if v >= 0.5 else 0.0

    return float(value)


@app.route("/")
def index():
    if _artifact_error:
        return render_template(
            "index.html",
            feature_names=[],
            raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
            suggested_defaults=SUGGESTED_DEFAULTS,
            region_options=[
                {"value": 0, "label": "Bejaia"},
                {"value": 1, "label": "Sidi-Bel Abbes"},
            ],
            error=_artifact_error,
            prediction=None,
        )
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
        suggested_defaults=SUGGESTED_DEFAULTS,
        region_options=[
            {"value": 0, "label": "Bejaia"},
            {"value": 1, "label": "Sidi-Bel Abbes"},
        ],
        error=None,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])  # form submission
def predict_form():
    if _artifact_error:
        return render_template(
            "index.html",
            feature_names=[],
            raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
            suggested_defaults=SUGGESTED_DEFAULTS,
            region_options=[
                {"value": 0, "label": "Bejaia"},
                {"value": 1, "label": "Sidi-Bel Abbes"},
            ],
            error=_artifact_error,
            prediction=None,
        ), 500

    try:
        raw_values_map = {}
        missing_fields = []
        for name in DISPLAY_RAW_FEATURE_NAMES:
            raw_val = request.form.get(name)
            if raw_val is None or raw_val == "":
                missing_fields.append(name)
            else:
                raw_values_map[name] = _coerce_raw_value(name, raw_val)

        if missing_fields:
            return render_template(
                "index.html",
                feature_names=FEATURE_NAMES,
                raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
                suggested_defaults=SUGGESTED_DEFAULTS,
                region_options=[
                    {"value": 0, "label": "Bejaia"},
                    {"value": 1, "label": "Sidi-Bel Abbes"},
                ],
                error=f"Missing inputs for: {', '.join(missing_fields)}",
                prediction=None,
            ), 400

        # Arrange values in the exact order expected by the scaler/model
        # Any raw features not used by the model (e.g., highly correlated ones)
        # will be ignored here because they won't be in FEATURE_NAMES.
        ordered_values = [raw_values_map[name] for name in FEATURE_NAMES]
        features = np.array(ordered_values, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)
        prediction_value = float(pred[0])
        return render_template(
            "index.html",
            feature_names=FEATURE_NAMES,
            raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
            suggested_defaults=SUGGESTED_DEFAULTS,
            region_options=[
                {"value": 0, "label": "Bejaia"},
                {"value": 1, "label": "Sidi-Bel Abbes"},
            ],
            error=None,
            prediction=prediction_value,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            feature_names=FEATURE_NAMES,
            raw_feature_names=DISPLAY_RAW_FEATURE_NAMES,
            suggested_defaults=SUGGESTED_DEFAULTS,
            region_options=[
                {"value": 0, "label": "Bejaia"},
                {"value": 1, "label": "Sidi-Bel Abbes"},
            ],
            error=f"Prediction error: {exc}",
            prediction=None,
        ), 500


@app.route("/api/predict", methods=["POST"])  # JSON API
def predict_api():
    if _artifact_error:
        return jsonify({"error": _artifact_error}), 500

    try:
        payload = request.get_json(silent=True) or {}
        # Accept either: {"features": [...]} already in model order, or
        # a dict keyed by raw feature names.
        if isinstance(payload.get("features"), list):
            ordered_values = [float(v) for v in payload["features"]]
        else:
            raw_map = {name: payload.get(name, None) for name in DISPLAY_RAW_FEATURE_NAMES}
            missing = [k for k, v in raw_map.items() if v is None]
            if missing:
                return jsonify({"error": "Missing inputs", "missing": missing}), 400
            coerced_map = {k: _coerce_raw_value(k, v) for k, v in raw_map.items()}
            ordered_values = [coerced_map[name] for name in FEATURE_NAMES]

        features = np.array(ordered_values, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)
        return jsonify({
            "prediction": float(pred[0]),
            "feature_order": FEATURE_NAMES,
            "raw_feature_names": DISPLAY_RAW_FEATURE_NAMES,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "n_features": len(FEATURE_NAMES),
        "error": _artifact_error,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
