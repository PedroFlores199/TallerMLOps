import argparse
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error
from taller.domain import FEATURES
from taller.registry import Registry
from monitoring.drift import assess
from .generate import generate
from .train import train


def run(data_folder, artifact_folder):
    data, artifacts = Path(data_folder), Path(artifact_folder)
    registry = Registry(artifacts / "registry")
    old_bundle, old_meta = registry.load()
    fixed_hash = sha256((data / "validation.parquet").read_bytes()).hexdigest()
    reference = pd.read_parquet(data / "train.parquet")
    heldout = generate(4000, seed=47, price_factor=1.2, offset=100000)
    before = old_bundle["model"].predict(heldout[FEATURES])
    before_error = float(mean_absolute_error(heldout.cost_eur, before[:, 0]))
    monitoring = assess(reference, heldout, artifacts / "reports")
    candidate = None
    if monitoring["retraining_requested"]:
        new_orders = generate(20000, seed=44, price_factor=1.2, offset=50000)
        path = data / "market-training.parquet"
        new_orders.to_parquet(path, index=False)
        assert not set(new_orders.order_id) & set(heldout.order_id)
        candidate = train(data, artifacts, extra_data=path)
    current, current_meta = registry.load()
    after = current["model"].predict(heldout[FEATURES])
    after_error = float(mean_absolute_error(heldout.cost_eur, after[:, 0]))
    assert sha256((data / "validation.parquet").read_bytes()).hexdigest() == fixed_hash
    report = {
        "synthetic": True,
        "scenario": "Parts component prices +20%; labor rates unchanged",
        "evaluation_rows": len(heldout),
        "heldout_seed": 47,
        "retraining_seed": 44,
        "heldout_not_used_for_training": True,
        "fixed_validation_unchanged": True,
        "fixed_validation_sha256": fixed_hash,
        "drift": monitoring,
        "before": {
            "version": old_meta["version"],
            "validation_mae": old_meta["validation_cost_mae"],
            "current_market_cost_mae": before_error,
        },
        "after": {
            "version": current_meta["version"],
            "validation_mae": current_meta["validation_cost_mae"],
            "current_market_cost_mae": after_error,
        },
        "promoted": bool(candidate and candidate["promoted"]),
        "improvement_pct": (before_error - after_error) / before_error * 100,
    }
    (artifacts / "market-experiment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    a = p.parse_args()
    run(a.data, a.artifacts)
