"""Config, seeding and JSON helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import load_config, load_json, project_root, save_json, set_seed


def test_project_root_and_config():
    cfg = load_config()
    assert (project_root() / "config.yaml").exists()
    assert len(cfg["universe"]["equities"]) == 10
    assert cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"] == 1.0


def test_json_roundtrip():
    path = Path("tests/_tmp_roundtrip.json")
    try:
        save_json({"a": 1, "b": [1, 2]}, path)
        assert load_json(path) == {"a": 1, "b": [1, 2]}
    finally:
        path.unlink(missing_ok=True)


def test_seed_is_deterministic():
    set_seed(7)
    a = np.random.normal(size=5)
    set_seed(7)
    b = np.random.normal(size=5)
    np.testing.assert_array_equal(a, b)
