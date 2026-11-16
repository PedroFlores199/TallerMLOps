FROM python:3.12-slim-trixie AS serving
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock && useradd -m app && mkdir /data && chown app:app /data
COPY taller ./taller
COPY pipelines ./pipelines
COPY monitoring ./monitoring
COPY data/demo ./artifacts
USER app
ENV ARTIFACT_DIR=/app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "taller.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM serving AS training
USER root
COPY requirements-training.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-training.lock && mkdir -p /workspace/artifacts /workspace/data && chown -R app:app /workspace
USER app
ENV MLFLOW_DISABLE_AGENT_HINT=1
CMD ["python", "-m", "pipelines.train", "--data", "/workspace/data", "--artifacts", "/workspace/artifacts"]
