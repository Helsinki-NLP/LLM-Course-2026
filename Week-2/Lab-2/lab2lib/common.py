from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    required = {"experiment_name", "seed", "backend", "data", "models", "decoding", "output_dir"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


def project_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def output_dir(config: dict[str, Any]) -> Path:
    return project_path(config["output_dir"], config)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def load_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return read_jsonl(project_path(config["data"][split], config))


def load_knowledge(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = read_jsonl(project_path(config["data"]["knowledge"], config))
    limit = config["data"].get("knowledge_limit")
    return records[: int(limit)] if limit is not None else records


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def config_digest(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def set_seed(seed: int, include_torch: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def result_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / "results.json"


def initialize_results(config: dict[str, Any]) -> dict[str, Any]:
    splits = {
        name: len(load_split(config, name))
        for name in ("train", "dev", "test")
    }
    splits["knowledge"] = len(load_knowledge(config))
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "config_sha256": config_digest(config),
        "reproducibility": {
            "seed": config["seed"],
            "backend": config["backend"],
            "decoding": config["decoding"],
            "requested_models": config["models"],
        },
        "data": {
            "splits": splits,
            "metric": "exact-match accuracy",
            "labels": config["task"]["labels"],
        },
        "config": public_config,
        "baselines": [],
        "artifacts": {},
    }
    write_json(result_path(config), result)
    return result


def read_results(config: dict[str, Any]) -> dict[str, Any]:
    path = result_path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def update_results(config: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    path = result_path(config)
    current = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else initialize_results(config)
    )
    current.update(updates)
    write_json(path, current)
    return current
