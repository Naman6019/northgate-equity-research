"""Northgate Quantitative Research - shared utilities."""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return ROOT


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path_from_config(cfg: Dict[str, Any], key: str) -> Path:
    return ROOT / cfg["paths"][key]


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("raw", "processed", "models", "reports", "figures"):
        path_from_config(cfg, key).mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["news"]["cache_dir"]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str = "northgate", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
