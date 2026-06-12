"""Pytest configuration for tests directory."""

import logging
import os
import sys
import types

import pytest

# Run the unit suite CPU-only by default. transformers eagerly initializes the
# CUDA device when a TrainingArguments-derived config is constructed, so config
# tests would otherwise crash on a host whose driver can't init the installed
# torch build. Set CUDA_VISIBLE_DEVICES explicitly to run GPU-marked tests.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# The ``vastai`` SDK makes a network call at import time.  A minimal stub
# is inserted before importing any server modules so that tests never
# trigger real network traffic.
if "vastai" not in sys.modules:
    _vastai = types.ModuleType("vastai")
    _vastai.VastAI = type("VastAI", (), {})  # type: ignore[attr-defined]
    sys.modules["vastai"] = _vastai
    for _sub in ("vastai.vastai_sdk", "vastai.vast"):
        sys.modules.setdefault(_sub, types.ModuleType(_sub))


def pytest_configure(config):
    """Configure pytest to show executor and connector logs in output."""
    config.addinivalue_line("markers", "gpu: requires GPU/CUDA hardware")

    # Set up logging to display DEBUG level logs to stdout
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(process)d] %(name)s - %(levelname)s - %(message)s",
    )

    # Ensure worker, executors, connectors and vllm loggers show appropriate levels
    logging.getLogger("worker").setLevel(logging.DEBUG)
    logging.getLogger("executors").setLevel(logging.DEBUG)
    logging.getLogger("connectors").setLevel(logging.DEBUG)
    logging.getLogger("vllm").setLevel(logging.INFO)

    # Ensure all loggers propagate to root (don't filter out)
    for logger_name in ["worker", "executors", "connectors", "vllm"]:
        logging.getLogger(logger_name).propagate = True


@pytest.fixture(autouse=True)
def log_test_info(caplog):
    """Capture logs for each test at DEBUG level."""
    caplog.set_level(logging.DEBUG)
    yield

    # Optionally print captured logs if needed
    if caplog.records:
        print("\n--- Captured logs ---")
        for record in caplog.records:
            print(f"[{record.name}] {record.levelname}: {record.getMessage()}")
        print("--- End captured logs ---\n")


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests with the asyncio backend only."""
    return "asyncio"
