# VLLM DeepStream Plugin

A GStreamer plugin for NVIDIA DeepStream that integrates Vision-Language Models (VLM) using VLLM for real-time video understanding and analysis.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Support](#model-support)

## Overview

The `nvvllmvlm` plugin enables integration of vision-language models (Cosmos-Reason2, etc.) into DeepStream pipelines. It processes video frames in configurable time segments and performs batch VLM inference asynchronously.

### Key Features

- **Segment-Based Processing**: Collects frames into time windows for batch inference
- **Flexible Frame Sampling**: FPS-based or interval-based subsampling
- **Async Inference**: Background worker thread for non-blocking processing
- **Multi-Stream Support**: One plugin instance handles multiple streams efficiently
- **Per-Stream Prompts**: Different prompts and settings for each stream
- **Configurable Models**: Supports video-native and image-only VLM models
- **GPU-Optimized**: Zero-copy GPU operations, shared model across streams
- **Flexible Input Formats**: PyTorch tensors, PIL Images, or numpy arrays

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA DeepStream SDK 9.0.0+
- Docker with NVIDIA Container Runtime
- Python 3.12+
- 40GB+ GPU memory required
- Currently supported on x86 based GPU platforms

## Quick Start

### Installation

```bash
# Launch DeepStream container
sudo docker run -it --rm --runtime=nvidia --gpus all --network=host \
  -v $(pwd):/home/vllm_ds_plugin \
  nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Inside DeepStream container install dependencies
cd /home/vllm_ds_plugin/deepstream-vllm-plugin
./install.sh
```

### Steps to get Hugging face Token

1. Log in at huggingface.co
2. Go to Profile  → Access Tokens
3. Create and save the generated token
4. To use cosmos-reason2 model, go to https://huggingface.co/nvidia/Cosmos-Reason2-8B, review and agree Nvidia Open Model License Agreement

### Single Stream and Multi-Stream Processing
```bash
# Run the following inside DeepStream container:

# Select GPU (optional)
export CUDA_VISIBLE_DEVICES=0

# Export Huggingface token from previous step to download models from HF
export HF_TOKEN=<Your HF Token>

# Copy streams to the container or use streams that are already part of the container

# Single stream (dry-run: results printed to console)
python3 vllm_ds_app_kafka_publish.py <video1.mp4 or RTSP stream url> --dry-run

For example,
python3 vllm_ds_app_kafka_publish.py /opt/nvidia/deepstream/deepstream-9.0/samples/streams/sample_1080p_h264.mp4 --dry-run

# Multi-stream with shared model (dry-run)
python3 vllm_ds_app_kafka_publish.py <video1.mp4 or RTSP url> <video2.mp4 or RTSP url> --dry-run
```

### Kafka Integration

Stream results to Kafka in real-time:
```bash
# Bring up kafka containers by running the following on host outside the DeepStream container:

# Start Kafka
docker compose -f docker-compose-kafka.yml up -d

# Run kafka publishing application and consumer script inside DeepStream container:

# Run with Kafka publishing (single or multi-stream)
python3 vllm_ds_app_kafka_publish.py <video1.mp4> <video2.mp4> \
    --kafka-bootstrap localhost:9092 --topic vlm-results

# On another terminal start the consumer test script
python3 test_consumer.py --topic vlm-results
```


## Configuration

### Configuration File (config.yaml)

Place `config.yaml` in the plugin directory or current working directory.

#### Complete Example

```yaml
# Model Configuration
model:
  path: "nvidia/Cosmos-Reason2-8B"
  max_model_len: 20480          # Max context length
  gpu_memory_utilization: 0.7   # GPU memory fraction to use. Update depending on platform and availabe gpu memory.
  trust_remote_code: true
  gpu_id: 0                     # GPU device ID (-1 for auto)

  # Video processing mode
  video_mode: 1  # 1=video metadata (native video support), 0=multi-image mode

  # Tensor format for image inputs
  tensor_format: "pytorch"  # pytorch/pil/numpy

# Segment Processing
segment:
  length_sec: 30                # Segment length in seconds
  overlap_sec: 0                # Segment overlap in seconds
  subsample_interval: 1         # Keep every Nth frame
  selection_fps: 30             # Target FPS (0 to disable FPS-based sampling)

# Inference Configuration
inference:
  # User prompt with optional placeholders: {num_frames}, {stream_id}, {timestamps}
  # Control timestamp inclusion by using or omitting {timestamps} placeholder
  user_prompt: "These are {num_frames} images from stream {stream_id} sampled at timestamps {timestamps}. Describe the scene in detail."

  # System prompt (optional - omit for no system prompt)
  system_prompt: |
    Provide captions with timestamps using format:
    <start time> <end time> caption of event.

  # Sampling parameters (all optional)
  max_tokens: 2048             # Max tokens to generate
  temperature: 0.7              # Sampling temperature (0-1)

  # Advanced sampling parameters (optional)
  # top_p: 0.9                  # Top-p nucleus sampling
  # top_k: 100                  # Top-k sampling
  # repetition_penalty: 1.1     # Repetition penalty

  # Per-stream prompt overrides (multi-stream mode)
  stream_prompts:
    0:  # Stream 0
      user_prompt: "Stream {stream_id} at {timestamps}: Monitor for security threats."
      system_prompt: "You are a security analyst."
    1:  # Stream 1
      user_prompt: "Analyze traffic flow."  # No {timestamps} = no timestamps

# Pipeline Configuration
pipeline:
  queue_maxsize: 20             # Max inference queue size
  max_wait_timeout: 300         # Max wait time for segment completion (seconds)

# Video Configuration
video:
  default_fps_numerator: 30     # Default FPS numerator (30/1 = 30 fps)
  default_fps_denominator: 1    # Default FPS denominator (used if stream lacks FPS)
```

#### Key Settings

**Video Mode**:
- `video_mode: 1` - Video metadata (For models that has native video support)
- `video_mode: 0` - Multi-image mode (image-only models)

**Tensor Format** (image modes only):
- `tensor_format: "pytorch"` - PyTorch tensors (default)
- `tensor_format: "pil"` - PIL Images
- `tensor_format: "numpy"` - numpy arrays

**System Prompt**:
- Specified: Uses that value
- Omitted: No system prompt (None)
- Empty string `""`: Sends empty system prompt

**Prompt Placeholders**:
- `{num_frames}` - Number of frames in segment
- `{stream_id}` - Stream identifier
- `{timestamps}` - Timestamp string (e.g., "0.00s 1.00s 2.00s")
- Include `{timestamps}` to show timestamps, omit to exclude them

**Per-Stream Prompts**:
- Override any inference setting for specific streams
- Streams without overrides use global defaults
- Supports all inference settings per-stream

#### Plugin Properties

Properties can override config values at runtime:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | string | from config | HuggingFace model ID |
| `user-prompt` | string | from config | User prompt with placeholders |
| `system-prompt` | string | None | System prompt (optional) |
| `segment-length-sec` | int | 10 | Segment length in seconds |
| `overlap-sec` | int | 0 | Segment overlap |
| `selection-fps` | int | 1 | Target FPS (0=disabled) |
| `subsample-interval` | int | 1 | Keep every Nth frame |
| `max-tokens` | int | 2048 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top-p` | float | 0.9 | Top-p nucleus sampling |
| `top-k` | int | 100 | Top-k sampling |
| `repetition-penalty` | float | 1.1 | Repetition penalty |
| `max-model-len` | int | 20480 | Max model context length |
| `trust-remote-code` | bool | true | Trust remote code |
| `gpu-memory-utilization` | float | 0.7 | GPU memory fraction (0.0-1.0) |
| `gpu-id` | int | 0 | GPU device ID (-1=auto) |
| `video-mode` | int | 1 | 1=video, 0=multi-image |
| `tensor-format` | string | pytorch | pytorch/pil/numpy |
| `queue-maxsize` | int | 20 | Inference queue size |
| `max-wait-timeout` | int | 300 | Shutdown timeout (seconds) |
| `default-fps-numerator` | int | 30 | Default FPS numerator |
| `default-fps-denominator` | int | 1 | Default FPS denominator |


## Model Support

### Supported Models

**Video-Native Models** (`video_mode: 1`):
- `Cosmos-Reason2-8B` (default)
- Models with native video metadata support

**Image-Only Models** (`video_mode: 0`):
- Models that only support image inputs

### Model Configuration Examples

**Cosmos-Reason2-8B**:
```yaml
model:
  path: "nvidia/Cosmos-Reason2-8B"
  video_mode: 1
  tensor_format: "pytorch"
```

### Custom Prompts with Placeholders

Control prompt content and timestamps using placeholders:

**With timestamps**:
```yaml
inference:
  user_prompt: "These are {num_frames} images from stream {stream_id} sampled at timestamps {timestamps}. Describe the scene."
```

**Without timestamps**:
```yaml
inference:
  user_prompt: "Describe what you see in stream {stream_id}."
```

**Per-stream custom prompts**:
```yaml
inference:
  stream_prompts:
    0:
      user_prompt: "Stream {stream_id} at {timestamps}: Security analysis."
    1:
      user_prompt: "Analyse vehicles."  # Minimal, no placeholders
```


### Signal-Based Results

Access results via GObject signals:
```python
def on_vlm_result(element, stream_id, start_time, end_time, text, user_data):
    print(f"Stream {stream_id} [{start_time:.2f}s-{end_time:.2f}s]: {text}")

vlm.connect("vlm-result", on_vlm_result, None)
```
