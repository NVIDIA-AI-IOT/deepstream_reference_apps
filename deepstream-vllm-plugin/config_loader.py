###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################################

"""Configuration loader for VLM plugin"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for VLM plugin"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file. If None, searches for
                        config.yaml in:
                        1. Current directory
                        2. Script directory
                        3. Uses default values
        """
        self._config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load config from file or use defaults"""

        # Try to find config file
        if config_path is None:
            # Search in common locations
            search_paths = [
                Path.cwd() / "config.yaml",
                Path(__file__).parent / "config.yaml",
            ]

            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break

        # Load from file if found
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        # Return empty dict if no config found (will use property defaults)
        return {}

    # Model properties
    @property
    def model_path(self) -> str:
        return self._config.get("model", {}).get(  # noqa: BLK100
            "path", "nvidia/Cosmos-Reason2-8B"
        )

    @property
    def max_model_len(self) -> int:
        return self._config.get("model", {}).get("max_model_len", 20480)

    @property
    def trust_remote_code(self) -> bool:
        return self._config.get("model", {}).get("trust_remote_code", True)

    @property
    def gpu_memory_utilization(self) -> float:
        return self._config.get("model", {}).get("gpu_memory_utilization", 0.4)

    @property
    def gpu_id(self) -> int:
        return self._config.get("model", {}).get("gpu_id", 0)

    @property
    def video_mode(self) -> int:
        return self._config.get("model", {}).get("video_mode", 1)

    @property
    def tensor_format(self) -> str:
        return self._config.get("model", {}).get("tensor_format", "pytorch")

    # Segment properties
    @property
    def segment_length_sec(self) -> int:
        return self._config.get("segment", {}).get("length_sec", 10)

    @property
    def overlap_sec(self) -> int:
        return self._config.get("segment", {}).get("overlap_sec", 0)

    @property
    def subsample_interval(self) -> int:
        return self._config.get("segment", {}).get("subsample_interval", 1)

    @property
    def selection_fps(self) -> int:
        return self._config.get("segment", {}).get("selection_fps", 1)

    # Inference properties
    @property
    def user_prompt(self) -> str:
        return self._config.get("inference", {}).get(
            "user_prompt", "Describe the scene in detail."
        )

    @property
    def system_prompt(self) -> str:
        # Return None if not specified in config (no default system prompt)
        return self._config.get("inference", {}).get("system_prompt", None)

    @property
    def max_tokens(self) -> int:
        """Max tokens default: 2048"""
        return self._config.get("inference", {}).get("max_tokens", 2048)

    @property
    def temperature(self) -> float:
        """Temperature default: 0.7"""
        return self._config.get("inference", {}).get("temperature", 0.7)

    @property
    def top_p(self) -> Optional[float]:
        """Top-p (nucleus) sampling parameter"""
        return self._config.get("inference", {}).get("top_p", None)

    @property
    def top_k(self) -> Optional[int]:
        """Top-k sampling parameter"""
        value = self._config.get("inference", {}).get("top_k", None)
        return int(value) if value is not None else None

    @property
    def repetition_penalty(self) -> Optional[float]:
        """Repetition penalty parameter"""
        return self._config.get("inference", {}).get(  # noqa: BLK100
            "repetition_penalty", None
        )

    @property
    def stream_prompts(self) -> dict:
        """
        Get per-stream prompt overrides
        Returns dict: {stream_id: {setting: value, ...}}
        """
        return self._config.get("inference", {}).get("stream_prompts", {})

    # Pipeline properties
    @property
    def queue_maxsize(self) -> int:
        return self._config.get("pipeline", {}).get("queue_maxsize", 20)

    @property
    def max_wait_timeout(self) -> int:
        return self._config.get("pipeline", {}).get("max_wait_timeout", 300)

    # Video properties
    @property
    def default_fps(self) -> tuple:
        numerator = self._config.get("video", {}).get(  # noqa: BLK100
            "default_fps_numerator", 30
        )
        denominator = self._config.get("video", {}).get(  # noqa: BLK100
            "default_fps_denominator", 1
        )
        return (numerator, denominator)


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
