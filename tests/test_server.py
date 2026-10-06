from unittest.mock import Mock

import pytest

import server

PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


@pytest.fixture
def client():
    with server.app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "model_version": server.MODEL_VERSION}


def test_predict_orders_features_and_returns_numeric_prediction(client, monkeypatch):
    predict = Mock(return_value=[123.45])
    monkeypatch.setattr(server.model, "predict", predict)

    response = client.post("/predict", json=dict(reversed(list(PAYLOAD.items()))))

    assert response.status_code == 200
    assert response.json == {"prediction": 123.45}
    predict.assert_called_once()
    features = predict.call_args.args[0]
    assert features.columns.tolist() == server.FEATURE_NAMES
    assert features.iloc[0].tolist() == [PAYLOAD[name] for name in server.FEATURE_NAMES]


def test_predict_rejects_missing_feature(client):
    payload = {name: value for name, value in PAYLOAD.items() if name != "s6"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json == {
        "error": "Invalid input",
        "details": {"s6": "Required field is missing"},
    }


@pytest.mark.parametrize("value", ["invalid", True, None, float("inf")])
def test_predict_rejects_invalid_number(client, value):
    response = client.post("/predict", json={**PAYLOAD, "age": value})
    assert response.status_code == 422
    assert response.json["error"] == "Invalid input"
    assert "age" in response.json["details"]


def test_predict_rejects_malformed_json(client):
    response = client.post("/predict", data='{"age":', content_type="application/json")
    assert response.status_code == 422
    assert response.json == {
        "error": "Invalid input",
        "details": {"body": "Expected a JSON object"},
    }
