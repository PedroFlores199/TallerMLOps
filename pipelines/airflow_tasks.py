import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from monitoring.drift import assess
from taller.registry import Registry
from .generate import generate, prepare
from .market_experiment import run
from .train import train


def execute(action, folder):
    folder = Path(folder)
    data, artifacts = folder / "data", folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if action == "prepare":
        if not (data / "manifest.json").exists():
            prepare(data)
        return json.loads((data / "manifest.json").read_text())
    if action == "baseline":
        current = Registry(artifacts / "registry").current()
        return current if current else train(data, artifacts)
    if action == "drift":
        factor = float(os.getenv("PRICE_FACTOR", "1.2"))
        if factor not in [1.0, 1.2]:
            raise ValueError("Escenario no admitido")
        reference = pd.read_parquet(data / "train.parquet")
        current = generate(4000, seed=47, price_factor=factor, offset=100000)
        return assess(reference, current, artifacts / "reports")
    if action == "retrain":
        return run(data, artifacts)
    if action == "record":
        current = Registry(artifacts / "registry").current()
        drift = json.loads((artifacts / "reports/drift-summary.json").read_text())
        result = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_version": current["version"],
            "drift_detected": drift["retraining_requested"],
            "synthetic": True,
        }
        (artifacts / "airflow-result.json").write_text(json.dumps(result, indent=2))
        return result
    raise ValueError("Acción no admitida")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "baseline", "drift", "retrain", "record"]
    )
    parser.add_argument("--folder", default="/shared")
    args = parser.parse_args()
    print(json.dumps(execute(args.action, args.folder), indent=2))
