from threading import RLock
import numpy as np
import pandas as pd
import shap
from .domain import FEATURES, CATEGORICAL
from .registry import Registry


class Estimator:
    def __init__(self, folder, version=None):
        self.version = version
        self.registry = Registry(folder)
        self.bundle = None
        self.metadata = None
        self.lock = RLock()
        self.explainer = None

    def refresh(self):
        current = (
            self.registry.version_metadata(self.version)
            if self.version
            else self.registry.current()
        )
        if not current:
            raise FileNotFoundError("No hay un modelo entrenado")
        with self.lock:
            if self.metadata is None or current["version"] != self.metadata["version"]:
                self.bundle, self.metadata = self.registry.load(current)
                self.explainer = shap.TreeExplainer(
                    self.bundle["model"].named_steps["regressor"].estimators_[0]
                )

    def predict(self, values):
        with self.lock:
            self.refresh()
            model = self.bundle["model"]
            frame = pd.DataFrame([{name: values[name] for name in FEATURES}])
            prediction = model.predict(frame)[0]
            scales = np.maximum(np.abs(prediction), [50.0, 0.5])
            radii = scales * self.bundle["quantiles"]
            transformed = model.named_steps["features"].transform(frame)
            shap_values = np.asarray(self.explainer.shap_values(transformed))[0]
            base = float(np.asarray(self.explainer.expected_value).reshape(-1)[0])
            names = model.named_steps["features"].get_feature_names_out()
            if not np.isclose(
                base + shap_values.sum(), prediction[0], rtol=1e-3, atol=0.1
            ):
                raise ValueError("La explicación no reproduce la predicción del modelo")
            grouped = {}
            for name, value in zip(names, shap_values):
                group = next(
                    (
                        feature
                        for feature in CATEGORICAL
                        if str(name).startswith(feature + "_")
                    ),
                    str(name),
                )
                grouped[group] = grouped.get(group, 0.0) + float(value)
            contributions = sorted(
                [
                    {
                        "feature": name,
                        "value": values[name],
                        "amount_eur": round(value, 2),
                    }
                    for name, value in grouped.items()
                ],
                key=lambda v: abs(v["amount_eur"]),
                reverse=True,
            )
            outside = [
                name
                for name, bounds in self.metadata.get("numeric_ranges", {}).items()
                if not bounds[0] <= float(values[name]) <= bounds[1]
            ]
            return {
                "cost_eur": round(max(0, float(prediction[0])), 2),
                "cost_interval": {
                    "lower": round(max(0, float(prediction[0] - radii[0])), 2),
                    "upper": round(max(0, float(prediction[0] + radii[0])), 2),
                    "nominal": 0.9,
                },
                "labor_hours": round(max(0, float(prediction[1])), 2),
                "hours_interval": {
                    "lower": round(max(0, float(prediction[1] - radii[1])), 2),
                    "upper": round(max(0, float(prediction[1] + radii[1])), 2),
                },
                "base_cost_eur": round(base, 2),
                "explanation": contributions[:7],
                "explanation_all_sum": round(float(shap_values.sum()), 2),
                "model_version": self.metadata["version"],
                "model": self.metadata["model"],
                "synthetic_training_data": True,
                "vat_included": 0.21,
                "out_of_distribution": outside,
                "interval_applicable_to_observed_domain": not outside,
                "scope": "Estimación de demostración con datos sintéticos; no es un presupuesto comercial.",
            }
