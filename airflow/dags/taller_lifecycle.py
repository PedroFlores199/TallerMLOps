from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from airflow.sdk import DAG, Param
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator


def choose_path():
    summary = json.loads(
        Path("/shared/artifacts/reports/drift-summary.json").read_text()
    )
    return "retrain_and_gate" if summary["retraining_requested"] else "skip_retraining"


with DAG(
    dag_id="taller_model_lifecycle",
    description="Deriva, reentrenamiento y promoción con validación fija",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule="@weekly",
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=15)},
    params={"price_factor": Param(1.2, type="number", enum=[1.0, 1.2])},
    tags=["mlops", "synthetic", "portfolio"],
) as dag:

    def task(identifier, action, **kwargs):
        return BashOperator(
            task_id=identifier,
            bash_command="/opt/taller-venv/bin/python -m pipelines.airflow_tasks "
            + action
            + " --folder /shared",
            append_env=True,
            env={
                "PYTHONPATH": "/opt/taller",
                "MLFLOW_DISABLE_AGENT_HINT": "1",
                "PRICE_FACTOR": "{{ params.price_factor }}",
            },
            execution_timeout=timedelta(minutes=10),
            **kwargs,
        )

    prepare = task("prepare_orders", "prepare")
    baseline = task("ensure_baseline_model", "baseline")
    drift = task("detect_drift", "drift")
    branch = BranchPythonOperator(
        task_id="decide_retraining", python_callable=choose_path
    )
    retrain = task("retrain_and_gate", "retrain")
    skip = EmptyOperator(task_id="skip_retraining")
    record = task("record_result", "record", trigger_rule="none_failed_min_one_success")
    prepare >> baseline >> drift >> branch >> [retrain, skip] >> record
