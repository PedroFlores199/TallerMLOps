FROM apache/airflow:3.4.0-python3.12
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && python -m venv /opt/taller-venv
COPY requirements-training.lock /tmp/taller-requirements.lock
RUN /opt/taller-venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/taller-requirements.lock && mkdir -p /shared /opt/taller && chown -R airflow:0 /shared /opt/taller
COPY --chown=airflow:0 taller /opt/taller/taller
COPY --chown=airflow:0 pipelines /opt/taller/pipelines
COPY --chown=airflow:0 monitoring /opt/taller/monitoring
COPY --chown=airflow:0 airflow/dags /opt/airflow/dags
USER airflow
