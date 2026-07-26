from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_shape():
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 28,
        "children": 0,
        "smoker": "yes",
        "region": "northeast",
    }

    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert body["risk_category"] in {"Low", "Medium", "High"}
    assert body["predicted_annual_charge"] > 0
    assert len(body["key_risk_factors"]) == 3


def test_predict_rejects_invalid_smoker_value():
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 28,
        "children": 0,
        "smoker": "maybe",
        "region": "northeast",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
