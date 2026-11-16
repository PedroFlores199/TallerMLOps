from datetime import datetime, timezone
import json
from pathlib import Path
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from taller.domain import FEATURES, CATEGORICAL, NUMERIC, TARGETS


def assess(reference, current, folder):
    if len(reference) < 200 or len(current) < 200:
        raise ValueError("Se necesitan al menos 200 observaciones por muestra")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    columns = FEATURES + TARGETS
    definition = DataDefinition(
        numerical_columns=NUMERIC + TARGETS, categorical_columns=CATEGORICAL
    )
    ref = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur = Dataset.from_pandas(current[columns], data_definition=definition)
    result = Report(
        [
            DataDriftPreset(columns=FEATURES, drift_share=0.2),
            ValueDrift(column="parts_price_index", method="ks", threshold=0.01),
        ]
    ).run(reference_data=ref, current_data=cur)
    values = result.dict()
    metric = next(
        m
        for m in values["metrics"]
        if m["metric_name"].startswith("ValueDrift(column=parts_price_index,method=ks")
    )
    pvalue = float(metric["value"])
    mean_ratio = float(
        current.parts_price_index.mean() / reference.parts_price_index.mean()
    )
    detected = pvalue < 0.01 and abs(mean_ratio - 1) > 0.1
    summary = {
        "at": datetime.now(timezone.utc).isoformat(),
        "critical_feature": "parts_price_index",
        "ks_pvalue": pvalue,
        "practical_mean_shift": mean_ratio - 1,
        "statistical_threshold": 0.01,
        "practical_threshold": 0.1,
        "retraining_requested": detected,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "synthetic": True,
    }
    result.save_html(str(folder / "evidently.html"))
    (folder / "evidently.json").write_text(json.dumps(values, default=float, indent=2))
    (folder / "drift-summary.json").write_text(json.dumps(summary, indent=2))
    return summary
