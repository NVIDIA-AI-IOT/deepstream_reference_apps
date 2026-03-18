#!/usr/bin/env bash
################################################################################
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
################################################################################

# Script to set up environment for VLLM DS plugin

set -e  # Exit immediately if a command exits with a non-zero status

# 1) Install Python dependencies from requirements.txt
# --ignore-installed forces reinstall, useful if system packages conflict
pip install -r requirements.txt --ignore-installed

# 2) Update apt package index to get latest package metadata
apt update

# 3) Install Python GObject Introspection bindings (used with GStreamer)
apt install -y python3-gi

# 4) Install GStreamer Python 3 bindings
apt install -y python3-gst-1.0

# 5) Install the GStreamer Python plugin loader
apt install -y gstreamer1.0-python3-plugin-loader

