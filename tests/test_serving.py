from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient

import taller.main as api
from taller.predict import Estimator
from taller.router import app as router, lane

REFERENCE = {
    "brand": "Seat",
    "model": "Ibiza",
    "year": 2018,
    "kilometers": 120000,
    "engine": "Gasolina",
    "repair": "Abolladura",
    "part": "Puerta",
    "severity": 2,
    "region": "Almería",
}


@pytest.fixture
def client(monkeypatch):
    folder = Path(__file__).resolve().parent.parent / "data/demo/registry"
    monkeypatch.setattr(api, "estimator", Estimator(folder))
    with TestClient(api.app) as client:
        yield client


def test_actual_model_contract_and_shap_additivity(client):
    response = client.post("/api/predict", json=REFERENCE)
    assert response.status_code == 200
    result = response.json()
    assert (
        result["cost_interval"]["lower"]
        <= result["cost_eur"]
        <= result["cost_interval"]["upper"]
    )
    assert (
        abs(
            result["base_cost_eur"] + result["explanation_all_sum"] - result["cost_eur"]
        )
        < 0.03
    )
    assert result["synthetic_training_data"]
    assert result["out_of_distribution"] == []
    assert all(
        item["feature"] in REFERENCE
        or item["feature"]
        in ["labor_rate_eur", "parts_price_index", "month", "symptom", "dtc"]
        for item in result["explanation"]
    )


def test_estimates_outside_observed_domain_are_flagged(client):
    response = client.post(
        "/api/predict", json={**REFERENCE, "year": 1990, "kilometers": 1400000}
    )
    assert response.status_code == 200
    result = response.json()
    assert {"year", "kilometers"} <= set(result["out_of_distribution"])
    assert not result["interval_applicable_to_observed_domain"]


def test_closed_catalog_and_ranges_are_enforced(client):
    for invalid in [
        {"model": "Inventado"},
        {"severity": 0},
        {"engine": "Vapor"},
        {"parts_price_index": 5},
    ]:
        assert (
            client.post("/api/predict", json={**REFERENCE, **invalid}).status_code
            == 422
        )


def test_canary_allocation_is_sticky_and_approximately_ten_percent():
    assignments = [lane(str(i)) for i in range(1000)]
    assert 70 <= assignments.count("challenger") <= 130
    assert assignments == [lane(str(i)) for i in range(1000)]


def test_canary_failure_falls_back_to_champion():
    identifier = next(str(i) for i in range(100) if lane(str(i)) == "challenger")
    calls = []

    class FakeClient:
        async def request(self, method, url, **kwargs):
            calls.append(url)
            return httpx.Response(
                503 if "challenger" in url else 200, json={"model_version": "champion"}
            )

    with TestClient(router) as client:
        router.state.client = FakeClient()
        response = client.post(
            "/api/predict", headers={"X-Request-ID": identifier}, json=REFERENCE
        )
    assert response.status_code == 200
    assert response.headers["X-Canary-Fallback"] == "true"
    assert response.headers["X-Model-Lane"] == "champion"
    assert len(calls) == 2
