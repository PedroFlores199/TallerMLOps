import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from taller.domain import BRANDS, BRAND_FACTOR, TASKS, PARTS, REGIONS


def generate(count=50000, seed=42, price_factor=1.0, offset=0):
    rng = np.random.default_rng(seed)
    rows = []
    brands = list(BRANDS)
    tasks = list(TASKS)
    parts = list(PARTS)
    regions = list(REGIONS)
    for index in range(count):
        brand = str(rng.choice(brands))
        repair = str(rng.choice(tasks))
        symptom, dtc, base_parts, base_hours = TASKS[repair]
        year = int(rng.integers(2004, 2027))
        kilometers = int(rng.gamma(3, 35000))
        month = int(rng.integers(1, 13))
        severity = int(rng.integers(1, 4))
        part = str(rng.choice(parts))
        region = str(rng.choice(regions))
        price_index = float(rng.normal(1.0, 0.035)) * price_factor
        age = 2026 - year
        season = 1.12 if repair == "Aire acondicionado" and month in [6, 7, 8] else 1.0
        body = repair in [
            "Abolladura",
            "Arañazo",
            "Pieza rota",
            "Cristal roto",
            "Corrosión",
            "Pintura desprendida",
            "Pieza ausente",
        ]
        hours = (
            base_hours
            * (1 + age * 0.008 + min(kilometers, 500000) / 1500000)
            * (1 + 0.3 * (severity - 1) if body else 1.0)
            * rng.lognormal(-0.002, 0.055)
        )
        parts_cost = (
            base_parts
            * BRAND_FACTOR[brand]
            * (PARTS[part] if body else 1.0)
            * price_index
            * season
            * rng.lognormal(-0.002, 0.07)
        )
        labor_rate = REGIONS[region] + float(rng.normal(0, 1.5))
        cost = (parts_cost + hours * labor_rate + 12.0) * 1.21
        row = {
            "order_id": offset + index,
            "brand": brand,
            "model": str(rng.choice(BRANDS[brand])),
            "year": year,
            "kilometers": min(kilometers, 1500000),
            "engine": str(rng.choice(["Gasolina", "Diésel", "Híbrido"])),
            "repair": repair,
            "symptom": symptom,
            "dtc": dtc,
            "month": month,
            "severity": severity,
            "part": part,
            "region": region,
            "labor_rate_eur": round(labor_rate, 2),
            "parts_price_index": round(price_index, 5),
            "cost_eur": round(cost, 2),
            "labor_hours": round(hours, 3),
            "parts_component_eur": round(parts_cost, 2),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_splits(frame):
    split = frame.order_id.map(
        lambda x: int(sha256(str(x).encode()).hexdigest()[:8], 16) % 10
    )
    return {
        "train": frame[split < 7].copy(),
        "calibration": frame[split == 7].copy(),
        "validation": frame[split == 8].copy(),
        "test": frame[split == 9].copy(),
    }


def stress_holdout(frame):
    result = frame.copy()
    mask = (result.order_id % 2) == 0
    result.loc[mask, "cost_eur"] += result.loc[mask, "parts_component_eur"] * 0.2 * 1.21
    result.loc[mask, "parts_price_index"] *= 1.2
    return result


def prepare(folder, count=50000, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame = generate(count, seed)
    frame.to_parquet(folder / "orders.parquet", index=False)
    parts = fixed_splits(frame)
    for name, data in parts.items():
        if name in ["validation", "test", "calibration"]:
            data = stress_holdout(data)
        data.to_parquet(folder / (name + ".parquet"), index=False)
    manifest = {
        "count": count,
        "seed": seed,
        "synthetic": True,
        "reference_year": 2026,
        "vat": 0.21,
        "holdout_design": "Fixed before training: half baseline prices, half +20% parts. Temporal market simulation, not real workshop records.",
        "splits": {k: len(v) for k, v in parts.items()},
        "sha256": {
            p.name: sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.parquet")
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="data")
    p.add_argument("--count", type=int, default=50000)
    args = p.parse_args()
    print(json.dumps(prepare(args.folder, args.count), indent=2))
