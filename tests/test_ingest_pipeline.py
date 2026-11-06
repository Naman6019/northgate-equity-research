"""Offline loaders, retry helper and the one-command pipeline CLI."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.ingest import _retry, load_raw
from src.pipeline import main as pipeline_main


def _cfg(cfg):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["raw"] = "tests/_tmp_ing/raw"
    base = Path("tests/_tmp_ing")
    (base / "raw").mkdir(parents=True, exist_ok=True)
    return cfg, base


def test_retry_then_success():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    assert _retry(flaky, attempts=3, base_sleep=0) == "ok"
    assert calls["n"] == 3


def test_load_raw_and_missing_file(cfg):
    cfg2, base = _cfg(cfg)
    try:
        for name in ("prices", "macro", "news"):
            pd.DataFrame({"x": [1]}).to_parquet(base / "raw" / f"{name}.parquet", index=False)
        out = load_raw(cfg2)
        assert set(out) == {"prices", "macro", "news"}
        (base / "raw" / "news.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            load_raw(cfg2)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_pipeline_main_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr("src.pipeline.stage_data", lambda cfg, offline=False: calls.append(("data", offline)))
    monkeypatch.setattr("src.pipeline.stage_train", lambda cfg: calls.append(("train", False)))
    assert pipeline_main(["data", "--offline"]) == 0
    assert pipeline_main(["train"]) == 0
    assert pipeline_main(["all"]) == 0
    assert calls == [("data", True), ("train", False), ("data", False), ("train", False)]
