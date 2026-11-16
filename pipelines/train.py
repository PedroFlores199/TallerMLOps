import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from importlib.metadata import version
import os
from pathlib import Path
import time
from uuid import uuid4
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from taller.domain import FEATURES, CATEGORICAL, NUMERIC, TARGETS
from taller.registry import Registry
from .generate import prepare


def estimator(name, quick=False):
    if name == "random_forest":
        base = RandomForestRegressor(
            n_estimators=35 if quick else 120,
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=4,
            random_state=42,
        )
    elif name == "xgboost":
        base = XGBRegressor(
            n_estimators=50 if quick else 300,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=4,
            random_state=42,
        )
    else:
        base = LGBMRegressor(
            n_estimators=50 if quick else 320,
            num_leaves=31,
            learning_rate=0.06,
            n_jobs=4,
            random_state=42,
            verbosity=-1,
        )
    preprocessing = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", preprocessing),
            ("regressor", MultiOutputRegressor(base, n_jobs=1)),
        ]
    )


def metrics(actual, prediction):
    return {
        "cost_mae": float(mean_absolute_error(actual[:, 0], prediction[:, 0])),
        "cost_rmse": float(mean_squared_error(actual[:, 0], prediction[:, 0]) ** 0.5),
        "hours_mae": float(mean_absolute_error(actual[:, 1], prediction[:, 1])),
    }


def train(data_folder, artifact_folder, extra_data=None, quick=False):
    started = time.perf_counter()
    data = Path(data_folder)
    artifacts = Path(artifact_folder)
    artifacts.mkdir(parents=True, exist_ok=True)
    if not (data / "manifest.json").exists():
        prepare(data)
    sets = {
        name: pd.read_parquet(data / (name + ".parquet"))
        for name in ["train", "calibration", "validation", "test"]
    }
    if extra_data:
        sets["train"] = pd.concat(
            [sets["train"], pd.read_parquet(extra_data)], ignore_index=True
        )
    validation_sha = sha256((data / "validation.parquet").read_bytes()).hexdigest()
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///" + str((artifacts / "mlflow.db").resolve()),
        )
    )
    if mlflow.get_experiment_by_name("TallerMLOps") is None:
        mlflow.create_experiment(
            "TallerMLOps",
            artifact_location=(artifacts / "mlflow-runs").resolve().as_uri(),
        )
    mlflow.set_experiment("TallerMLOps")
    reports = []
    trained = {}
    for name in ["random_forest", "xgboost", "lightgbm"]:
        with mlflow.start_run(
            run_name=name + ("-market-shift" if extra_data else "-baseline")
        ) as run:
            model = estimator(name, quick)
            tick = time.perf_counter()
            model.fit(sets["train"][FEATURES], sets["train"][TARGETS].to_numpy())
            validation = model.predict(sets["validation"][FEATURES])
            score = metrics(sets["validation"][TARGETS].to_numpy(), validation)
            mlflow.log_params(
                {
                    "candidate": name,
                    "train_rows": len(sets["train"]),
                    "validation_sha256": validation_sha,
                    "synthetic": True,
                    "quick": quick,
                    "market_shift_training": bool(extra_data),
                }
            )
            mlflow.log_metrics(score)
            signature = infer_signature(
                sets["validation"][FEATURES].head(2),
                model.predict(sets["validation"][FEATURES].head(2)),
            )
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=signature,
                input_example=sets["validation"][FEATURES].head(2),
                registered_model_name="TallerRepairEstimator",
                pip_requirements=[
                    f"{package}=={version(package)}"
                    for package in ["scikit-learn", "pandas", "xgboost", "lightgbm"]
                ],
            )
            report = {
                "name": name,
                "run_id": run.info.run_id,
                "model_uri": info.model_uri,
                "seconds": round(time.perf_counter() - tick, 2),
                **score,
            }
            reports.append(report)
            trained[name] = model
            print(json.dumps(report), flush=True)
    winner = min(reports, key=lambda r: r["cost_mae"])
    model = trained[winner["name"]]
    calibration_prediction = model.predict(sets["calibration"][FEATURES])
    actual = sets["calibration"][TARGETS].to_numpy()
    scales = np.maximum(np.abs(calibration_prediction), np.array([50.0, 0.5]))
    scores = np.abs(actual - calibration_prediction) / scales
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * 0.9) / len(scores))
    quantiles = np.quantile(scores, quantile_level, axis=0, method="higher")
    test_prediction = model.predict(sets["test"][FEATURES])
    test_actual = sets["test"][TARGETS].to_numpy()
    test_score = metrics(test_actual, test_prediction)
    coverage = (
        np.abs(test_actual - test_prediction)
        <= np.maximum(np.abs(test_prediction), [50.0, 0.5]) * quantiles
    ).mean(axis=0)
    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid4().hex[:8],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": winner["name"],
        "mlflow_run_id": winner["run_id"],
        "validation_cost_mae": winner["cost_mae"],
        "validation_sha256": validation_sha,
        "test_metrics": test_score,
        "test_coverage": {"cost": float(coverage[0]), "hours": float(coverage[1])},
        "interval_nominal": 0.9,
        "quantiles": quantiles.tolist(),
        "train_rows": len(sets["train"]),
        "synthetic": True,
        "features": FEATURES,
        "candidates": reports,
    }
    metadata["numeric_ranges"] = {
        name: [float(sets["train"][name].min()), float(sets["train"][name].max())]
        for name in NUMERIC
    }
    registry = Registry(artifacts / "registry")
    metadata = registry.publish(
        {"model": model, "quantiles": quantiles, "features": FEATURES}, metadata
    )
    promoted = registry.promote(metadata)
    metadata["promoted"] = promoted
    metadata["pipeline_seconds"] = round(time.perf_counter() - started, 2)
    (artifacts / "latest-training.json").write_text(json.dumps(metadata, indent=2))
    if promoted:
        client = MlflowClient()
        versions = client.search_model_versions("name='TallerRepairEstimator'")
        match = next(v for v in versions if v.run_id == winner["run_id"])
        client.set_registered_model_alias(
            "TallerRepairEstimator", "champion", match.version
        )
    print(
        json.dumps(
            {
                "winner": winner["name"],
                "promoted": promoted,
                "version": metadata["version"],
                "test": test_score,
                "coverage": metadata["test_coverage"],
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--extra-data")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    train(a.data, a.artifacts, a.extra_data, a.quick)
