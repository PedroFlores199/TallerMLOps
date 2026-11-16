from datetime import date
import json
import os
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, model_validator
from .domain import BRANDS, TASKS, PARTS, REGIONS
from .predict import Estimator

ARTIFACTS = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
estimator = Estimator(ARTIFACTS / "registry", os.getenv("MODEL_VERSION"))
REQUESTS = Counter("taller_predictions_total", "Predicciones emitidas")
LATENCY = Histogram("taller_prediction_seconds", "Duración de predicción")
app = FastAPI(title="TallerMLOps", version="1.0.0")
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Repair(BaseModel):
    brand: str
    model: str
    year: int = Field(ge=1990, le=2027)
    kilometers: int = Field(ge=0, le=1500000)
    engine: str = "Gasolina"
    repair: str
    severity: int = Field(default=2, ge=1, le=3)
    part: str = "Sin determinar"
    month: int = Field(default_factory=lambda: date.today().month, ge=1, le=12)
    region: str = "Almería"
    parts_price_index: float = Field(default=1.0, ge=0.7, le=1.5)

    @model_validator(mode="after")
    def check(self):
        if self.brand not in BRANDS or self.model not in BRANDS[self.brand]:
            raise ValueError(
                "Marca y modelo no coinciden con el catálogo de demostración"
            )
        if (
            self.repair not in TASKS
            or self.part not in PARTS
            or self.region not in REGIONS
        ):
            raise ValueError("Reparación, pieza o provincia desconocida")
        if self.engine not in ["Gasolina", "Diésel", "Híbrido"]:
            raise ValueError("Motorización no admitida")
        if self.year > date.today().year:
            raise ValueError("El año del vehículo está en el futuro")
        return self

    def features(self):
        symptom, dtc, _, _ = TASKS[self.repair]
        return {
            **self.model_dump(),
            "symptom": symptom,
            "dtc": dtc,
            "labor_rate_eur": REGIONS[self.region],
        }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    try:
        estimator.refresh()
        return {"status": "ok", "version": estimator.metadata["version"]}
    except FileNotFoundError:
        raise HTTPException(503, "Modelo todavía no disponible")


@app.get("/api/catalog")
def catalog():
    return {
        "brands": BRANDS,
        "repairs": list(TASKS),
        "parts": list(PARTS),
        "regions": list(REGIONS),
    }


@app.get("/api/model")
def model():
    estimator.refresh()
    return estimator.metadata


@app.post("/api/predict")
async def predict(body: Repair):
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(estimator.predict, body.features())
    except FileNotFoundError:
        raise HTTPException(503, "Modelo todavía no disponible")
    REQUESTS.inc()
    LATENCY.observe(time.perf_counter() - started)
    return {**result, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/reports/market")
def market_report():
    path = ARTIFACTS / "market-experiment.json"
    if not path.exists():
        raise HTTPException(404, "El experimento aún no se ha ejecutado")
    return json.loads(path.read_text())


@app.get("/api/reports/evidently", include_in_schema=False)
def evidently_report():
    path = ARTIFACTS / "reports/evidently.html"
    if not path.exists():
        raise HTTPException(404, "No hay informe de deriva disponible")
    return FileResponse(path)
