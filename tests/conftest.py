"""Shared fixtures for immich-accelerator tests."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Override DATA_DIR / CONFIG_FILE / PID_DIR / LOG_DIR to use a temp directory."""
    data_dir = tmp_path / ".immich-accelerator"
    data_dir.mkdir()
    pid_dir = data_dir / "pids"
    pid_dir.mkdir()
    log_dir = data_dir / "logs"
    log_dir.mkdir()
    config_file = data_dir / "config.json"

    with patch.multiple(
        "immich_accelerator.__main__",
        DATA_DIR=data_dir,
        CONFIG_FILE=config_file,
        PID_DIR=pid_dir,
        LOG_DIR=log_dir,
    ):
        yield {
            "data_dir": data_dir,
            "config_file": config_file,
            "pid_dir": pid_dir,
            "log_dir": log_dir,
        }


@pytest.fixture
def sample_config():
    """A realistic config dict."""
    return {
        "version": "2.6.3",
        "server_dir": "/Users/test/.immich-accelerator/server/2.6.3",
        "node": "/opt/homebrew/bin/node",
        "db_hostname": "localhost",
        "db_port": "5432",
        "db_username": "postgres",
        "db_password": "secret",
        "db_name": "immich",
        "redis_hostname": "localhost",
        "redis_port": "6379",
        "upload_mount": "/Volumes/photos/upload",
        "ffmpeg_path": "/opt/homebrew/bin/ffmpeg",
        "ml_dir": "/Users/test/immich-ml-metal",
        "ml_port": 3003,
        "api_key": "test-api-key-123",
        "immich_url": "http://localhost:2283",
    }


@pytest.fixture
def saved_config(tmp_data_dir, sample_config):
    """Write sample_config to the temp config file and return it."""
    config_file = tmp_data_dir["config_file"]
    config_file.write_text(json.dumps(sample_config, indent=2))
    os.chmod(config_file, 0o600)
    return sample_config
