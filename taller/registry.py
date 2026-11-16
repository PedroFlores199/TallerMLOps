from hashlib import sha256
import json
from pathlib import Path
import re
from filelock import FileLock
import joblib


class Registry:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.pointer = self.folder / "production.json"

    def version_metadata(self, version):
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", version):
            raise ValueError("Invalid model version")
        data = json.loads((self.folder / version / "metadata.json").read_text())
        if data["version"] != version:
            raise ValueError("Model version mismatch")
        return data

    def current(self):
        return json.loads(self.pointer.read_text()) if self.pointer.exists() else None

    def publish(self, bundle, metadata):
        version = metadata["version"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", version):
            raise ValueError("Invalid model version")
        path = self.folder / version
        path.mkdir(exist_ok=False)
        artifact = path / "bundle.joblib"
        joblib.dump(bundle, artifact, compress=3)
        metadata = {**metadata, "sha256": sha256(artifact.read_bytes()).hexdigest()}
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    def promote(self, metadata):
        published = self.version_metadata(metadata["version"])
        if published != metadata:
            raise ValueError("Only published model metadata can be promoted")
        artifact = self.folder / metadata["version"] / "bundle.joblib"
        if sha256(artifact.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        with FileLock(str(self.folder / "production.lock")):
            current = self.current()
            if (
                current
                and metadata["validation_sha256"] != current["validation_sha256"]
            ):
                raise ValueError("The fixed validation set changed")
            if (
                current
                and metadata["validation_cost_mae"]
                >= current["validation_cost_mae"] * 0.995
            ):
                return False
            temporary = self.pointer.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, indent=2))
            temporary.replace(self.pointer)
            return True

    def load(self, metadata=None):
        metadata = metadata or self.current()
        if not metadata:
            raise FileNotFoundError("No trained model available")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", metadata["version"]):
            raise ValueError("Invalid model version")
        artifact = self.folder / metadata["version"] / "bundle.joblib"
        if sha256(artifact.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        return joblib.load(artifact), metadata
