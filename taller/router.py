from contextlib import asynccontextmanager
from hashlib import sha256
import os
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

CHAMPION = os.getenv("CHAMPION_URL", "http://champion:8000")
CHALLENGER = os.getenv("CHALLENGER_URL", "http://challenger:8000")
WEIGHT = int(os.getenv("CHALLENGER_PERCENT", "10"))
if not 0 <= WEIGHT <= 100:
    raise ValueError("El porcentaje debe estar entre cero y cien")


def lane(identifier, weight=WEIGHT):
    return (
        "challenger"
        if int(sha256(identifier.encode()).hexdigest()[:8], 16) % 100 < weight
        else "champion"
    )


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=10) as client:
        app.state.client = client
        yield


app = FastAPI(title="TallerMLOps traffic router", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "challenger_percent": WEIGHT}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    if path not in [
        "",
        "api/predict",
        "api/catalog",
        "api/model",
        "api/reports/market",
        "api/reports/evidently",
    ] and not path.startswith("static/"):
        return JSONResponse({"detail": "Ruta no disponible"}, status_code=404)
    identifier = request.headers.get("X-Request-ID", str(uuid4()))[:128]
    selected = lane(identifier) if path == "api/predict" else "champion"
    targets = (
        [selected, "champion", "champion"]
        if selected == "challenger"
        else ["champion", "champion"]
    )
    body = await request.body()
    if len(body) > 20000:
        return JSONResponse({"detail": "Solicitud demasiado grande"}, status_code=413)
    for target in targets:
        try:
            upstream = CHALLENGER if target == "challenger" else CHAMPION
            result = await request.app.state.client.request(
                request.method,
                upstream + "/" + path,
                params=request.query_params,
                content=body,
                headers={
                    "content-type": request.headers.get(
                        "content-type", "application/json"
                    )
                },
            )
            if result.status_code >= 500:
                continue
            return Response(
                result.content,
                status_code=result.status_code,
                headers={
                    "Content-Type": result.headers.get(
                        "content-type", "application/json"
                    ),
                    "X-Model-Lane": target,
                    "X-Request-ID": identifier,
                    "X-Canary-Fallback": str(target != selected).lower(),
                },
            )
        except httpx.HTTPError:
            continue
    return JSONResponse(
        {"detail": "Servicio temporalmente no disponible"}, status_code=503
    )
