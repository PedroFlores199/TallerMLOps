import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Event, Thread
import time

import httpx

ROOT = Path(__file__).resolve().parent.parent
BODY = {
    "brand": "Seat",
    "model": "Ibiza",
    "year": 2018,
    "kilometers": 120000,
    "engine": "Gasolina",
    "repair": "Abolladura",
    "part": "Puerta",
    "severity": 2,
    "region": "Almería",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs/kind-validation.json"
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--requests", type=int, default=500)
    args = parser.parse_args()
    if args.kubeconfig.exists():
        raise SystemExit("Usa un kubeconfig nuevo y separado del contexto habitual")
    env = {**os.environ, "KUBECONFIG": str(args.kubeconfig.resolve())}
    kind = shutil.which("kind")
    kubectl = shutil.which("kubectl")
    if not kind or not kubectl:
        raise SystemExit("Se necesitan Docker, Kind y kubectl")

    def command(*parts, **kwargs):
        return subprocess.run(list(parts), env=env, check=True, text=True, **kwargs)

    config = {
        "kind": "Cluster",
        "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [
            {
                "role": "control-plane",
                "extraPortMappings": [
                    {
                        "containerPort": 31085,
                        "hostPort": 5110,
                        "listenAddress": "127.0.0.1",
                        "protocol": "TCP",
                    }
                ],
            }
        ],
    }
    args.kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.kubeconfig.with_suffix(".kind.json")
    config_path.write_text(json.dumps(config))
    created = False
    try:
        command(
            kind,
            "create",
            "cluster",
            "--name",
            "taller-mlops",
            "--config",
            str(config_path),
            "--wait",
            "180s",
        )
        created = True
        command(
            kind, "load", "docker-image", "taller-mlops:demo", "--name", "taller-mlops"
        )
        command(kubectl, "apply", "-f", str(ROOT / "k8s/deployment.yaml"))
        for deployment in ["champion", "challenger", "router"]:
            command(
                kubectl,
                "-n",
                "taller",
                "rollout",
                "status",
                "deployment/" + deployment,
                "--timeout=240s",
            )
        counts, versions = Counter(), Counter()
        fallbacks = 0
        warmup = Counter()
        failures = []
        with httpx.Client(timeout=15) as client:
            deadline = time.monotonic() + 90
            index = 0
            while len(warmup) < 2 and time.monotonic() < deadline:
                response = client.post(
                    "http://127.0.0.1:5110/api/predict",
                    json=BODY,
                    headers={"X-Request-ID": "warmup-" + str(index)},
                )
                if (
                    response.status_code == 200
                    and response.headers.get("X-Canary-Fallback") == "false"
                ):
                    warmup[response.headers["X-Model-Lane"]] += 1
                index += 1
            if len(warmup) != 2:
                raise AssertionError(
                    "Las dos versiones no han alcanzado disponibilidad antes de la medición"
                )
            for index in range(args.requests):
                response = client.post(
                    "http://127.0.0.1:5110/api/predict",
                    json=BODY,
                    headers={"X-Request-ID": str(index)},
                )
                response.raise_for_status()
                counts[response.headers["X-Model-Lane"]] += 1
                versions[response.json()["model_version"]] += 1
                fallbacks += response.headers.get("X-Canary-Fallback") == "true"
        if (
            len(versions) != 2
            or not 0.05 <= counts["challenger"] / args.requests <= 0.15
        ):
            raise AssertionError(
                "No se han observado ambas versiones en el reparto real"
            )
        stopped = Event()
        successful = []

        def requests_during_rollout():
            with httpx.Client(timeout=15) as client:
                index = 1000
                while not stopped.is_set():
                    try:
                        response = client.post(
                            "http://127.0.0.1:5110/api/predict",
                            json=BODY,
                            headers={"X-Request-ID": str(index)},
                        )
                        response.raise_for_status()
                        successful.append(response.json()["model_version"])
                    except Exception as error:
                        failures.append(str(error))
                    index += 1
                    time.sleep(0.1)

        worker = Thread(target=requests_during_rollout, daemon=True)
        worker.start()
        try:
            command(
                kubectl,
                "-n",
                "taller",
                "set",
                "env",
                "deployment/champion",
                "MODEL_VERSION=20261115T222848_1f945185",
            )
            command(
                kubectl,
                "-n",
                "taller",
                "rollout",
                "status",
                "deployment/champion",
                "--timeout=240s",
            )
        finally:
            stopped.set()
            worker.join(timeout=30)
        state = command(
            kubectl, "-n", "taller", "get", "pods", "-o", "json", capture_output=True
        ).stdout
        report = {
            "at": datetime.now(timezone.utc).isoformat(),
            "environment": "Kind on local Docker",
            "canary_requests": args.requests,
            "warmup_lanes": dict(warmup),
            "canary_fallbacks": fallbacks,
            "lanes": dict(counts),
            "versions": dict(versions),
            "rolling_successful_requests": len(successful),
            "rolling_errors": failures,
            "pods": [
                {
                    "name": item["metadata"]["name"],
                    "namespace": item["metadata"]["namespace"],
                    "containers": [
                        {
                            "name": c["name"],
                            "restarts": c["restartCount"],
                            "ready": c["ready"],
                        }
                        for c in item.get("status", {}).get("containerStatuses", [])
                    ],
                }
                for item in json.loads(state)["items"]
            ],
            "scope": "Real local HTTP test after service warm-up, not a guarantee for production traffic.",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if failures or len(successful) < 10 or fallbacks:
            raise AssertionError(
                f"Rolling update: {len(failures)} errores / {len(successful)} respuestas correctas"
            )
    finally:
        if created and not args.keep:
            command(kind, "delete", "cluster", "--name", "taller-mlops")


if __name__ == "__main__":
    main()
