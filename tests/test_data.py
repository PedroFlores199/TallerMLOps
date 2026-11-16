import numpy as np
import pytest
from pipelines.generate import generate, fixed_splits, stress_holdout
from taller.domain import BRANDS, FEATURES, TARGETS
from taller.registry import Registry


def test_generator_is_reproducible_and_preserves_brand_model():
    a = generate(200, 42)
    b = generate(200, 42)
    assert a.equals(b)
    assert all(row.model in BRANDS[row.brand] for row in a.itertuples())
    assert (a.cost_eur > 0).all() and (a.labor_hours > 0).all()


def test_target_components_do_not_leak_into_features():
    assert not set(TARGETS + ["parts_component_eur", "order_id"]) & set(FEATURES)


def test_holdout_ids_are_disjoint_and_cover_all_orders():
    frame = generate(1000)
    splits = fixed_splits(frame)
    seen = set()
    for data in splits.values():
        ids = set(data.order_id)
        assert not ids & seen
        seen |= ids
    assert len(seen) == len(frame)


def test_twenty_percent_parts_shock_is_not_twenty_percent_total_bill():
    frame = generate(20)
    shifted = stress_holdout(frame)
    mask = frame.order_id % 2 == 0
    assert np.allclose(
        shifted.loc[mask, "parts_price_index"],
        frame.loc[mask, "parts_price_index"] * 1.2,
    )
    assert np.allclose(
        shifted.loc[mask, "cost_eur"] - frame.loc[mask, "cost_eur"],
        frame.loc[mask, "parts_component_eur"] * 0.2 * 1.21,
    )
    assert np.allclose(shifted.loc[~mask, "cost_eur"], frame.loc[~mask, "cost_eur"])


def test_registry_rejects_regression_and_changed_validation_set(tmp_path):
    registry = Registry(tmp_path)
    first = registry.publish(
        {"data": 1},
        {
            "version": "first-model",
            "validation_sha256": "fixed",
            "validation_cost_mae": 20.0,
        },
    )
    assert registry.promote(first)
    regression = registry.publish(
        {"data": 2},
        {
            "version": "regression-model",
            "validation_sha256": "fixed",
            "validation_cost_mae": 21.0,
        },
    )
    assert not registry.promote(regression)
    assert registry.current()["version"] == "first-model"
    changed = registry.publish(
        {"data": 3},
        {
            "version": "changed-validation",
            "validation_sha256": "changed",
            "validation_cost_mae": 10.0,
        },
    )
    with pytest.raises(ValueError):
        registry.promote(changed)


def test_registry_detects_tampered_artifact(tmp_path):
    registry = Registry(tmp_path)
    metadata = registry.publish({"data": 1}, {"version": "test-artifact"})
    (tmp_path / "test-artifact/bundle.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        registry.load(metadata)
