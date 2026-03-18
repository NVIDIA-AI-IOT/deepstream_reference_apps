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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
import multiprocessing as mp  # noqa: E402
import threading  # noqa: E402
from queue import Empty, Queue  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import torch  # noqa: E402
from gi.repository import GObject, Gst, GstBase  # noqa: E402
from PIL import Image  # noqa: E402
from pyservicemaker import Buffer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Load configuration
from config_loader import get_config  # noqa: E402

Gst.init(None)

GST_PLUGIN_NAME = "nvvllmvlm"

# Load config instance
config = get_config()

# Configuration values (can be overridden via config.yaml)
MODEL_PATH = config.model_path
DEFAULT_SEGMENT_LEN_SEC = config.segment_length_sec
DEFAULT_OVERLAP_SEC = config.overlap_sec
DEFAULT_SUBSAMPLE_INTERVAL = config.subsample_interval
DEFAULT_SELECTION_FPS = config.selection_fps


class BufferData:
    """Frame data for VLM processing"""

    def __init__(
        self,
        frame_number: int,
        pts: int,
        dts: int,
        duration: int,
        tensor_gpu: torch.Tensor,
    ) -> None:
        self.frame_number = frame_number
        self.pts = pts
        self.tensor_gpu = tensor_gpu


class Segment:
    """Temporal segment containing multiple frames"""

    def __init__(
        self, stream_id: int, start_pts_ns: int, end_pts_ns: int, batch_id: int
    ) -> None:
        self.stream_id = stream_id  # Which stream this segment belongs to
        self.start_pts_ns = start_pts_ns
        self.end_pts_ns = end_pts_ns
        self.batch_id = batch_id
        self.frames: List[BufferData] = []
        self.last_saved_pts_ns: Optional[int] = None


class StreamContext:
    """Per-stream context for multi-stream processing"""

    def __init__(self, stream_id: int):
        self.stream_id = stream_id

        # Segment management
        self.open_segments: List[Segment] = []
        self.next_segment_start_pts: Optional[int] = None
        self.base_pts: Optional[int] = None
        self.frame_counter: int = 0

        # Statistics
        self.segments_submitted: int = 0
        self.segments_completed: int = 0
        self.segments_dropped: int = 0
        self.total_frames_in_segments: int = 0

        # Latest result
        self.latest_text: Optional[str] = None
        self.latest_lock: threading.Lock = threading.Lock()

    def update_result(self, text: str, start_sec: float, end_sec: float):
        """Update the latest result for this stream"""
        with self.latest_lock:
            self.latest_text = (
                f"[Stream {self.stream_id}] "
                f"[{start_sec:.2f}s-{end_sec:.2f}s] {text}"
            )
            self.segments_completed += 1


class SegmentRequest:
    """Request for VLM inference on a segment"""

    def __init__(self, stream_id: int, segment: Segment, prompt_config: Dict):
        self.stream_id = stream_id
        self.segment = segment
        self.prompt_config = prompt_config


class NvVllmVLM(GstBase.BaseTransform):
    __gstmetadata__ = (
        "NvVllmVLM",
        "Generic/Analyzer",
        "vLLM inference with multi-stream support",
        "VSS",
    )

    src_format = Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGB, "
        "width=(int)[ 1, 2147483647 ], height=(int)[ 1, 2147483647 ], "
        "framerate=(fraction)[ 0/1, 2147483647/1 ]"
    )
    sink_format = Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGB, "
        "width=(int)[ 1, 2147483647 ], height=(int)[ 1, 2147483647 ], "
        "framerate=(fraction)[ 0/1, 2147483647/1 ]"
    )

    src_pad_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, src_format
    )
    sink_pad_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, sink_format
    )
    __gsttemplates__ = (src_pad_template, sink_pad_template)

    __gsignals__ = {
        "vlm-result": (
            GObject.SignalFlags.RUN_LAST,
            None,
            (
                int,
                float,
                float,
                str,
            ),
        )
        # Signal emitted when VLM inference completes for a segment
        # Args: stream_id (int), start_time (float), end_time (float),
        # result_text (str)
    }

    __gproperties__ = {
        "segment-length-sec": (
            int,
            "Segment Length (sec)",
            "Length of each segment in seconds",
            1,
            3600,
            DEFAULT_SEGMENT_LEN_SEC,
            GObject.ParamFlags.READWRITE,
        ),
        "overlap-sec": (
            int,
            "Overlap (sec)",
            "Overlap between consecutive segments",
            -600,
            600,
            DEFAULT_OVERLAP_SEC,
            GObject.ParamFlags.READWRITE,
        ),
        "subsample-interval": (
            int,
            "Subsample Interval",
            "Keep every Nth frame",
            1,
            100,
            DEFAULT_SUBSAMPLE_INTERVAL,
            GObject.ParamFlags.READWRITE,
        ),
        "selection-fps": (
            int,
            "Selection FPS",
            "Target frames per second per segment (0 = disabled)",
            0,
            240,
            DEFAULT_SELECTION_FPS,
            GObject.ParamFlags.READWRITE,
        ),
        "model": (
            str,
            "Model",
            "HuggingFace model id",
            MODEL_PATH,
            GObject.ParamFlags.READWRITE,
        ),
        "user-prompt": (
            str,
            "User Prompt",
            "User prompt with optional placeholders: "
            "{num_frames}, {stream_id}, {timestamps}",
            "Describe what you see in detail",
            GObject.ParamFlags.READWRITE,
        ),
        "max-tokens": (
            int,
            "Max Tokens",
            "Maximum tokens to generate",
            1,
            8192,
            2048,
            GObject.ParamFlags.READWRITE,
        ),
        "temperature": (
            float,
            "Temperature",
            "Sampling temperature",
            0.0,
            2.0,
            0.7,
            GObject.ParamFlags.READWRITE,
        ),
        "gpu-id": (
            int,
            "GPU ID",
            (
                "GPU device ID to use "
                "(default: 0, -1 = auto from CUDA_VISIBLE_DEVICES)"
            ),
            -1,
            15,
            0,
            GObject.ParamFlags.READWRITE,
        ),
        "video-mode": (
            int,
            "Video Mode",
            "Video mode (1=video metadata, 0=image-only)",
            0,
            1,
            1,
            GObject.ParamFlags.READWRITE,
        ),
        "tensor-format": (
            str,
            "Tensor Format",
            "Image tensor format: pytorch, pil, or numpy",
            "pytorch",
            GObject.ParamFlags.READWRITE,
        ),
        "top-p": (
            float,
            "Top P",
            "Top-p nucleus sampling",
            0.0,
            1.0,
            0.9,
            GObject.ParamFlags.READWRITE,
        ),
        "top-k": (
            int,
            "Top K",
            "Top-k sampling",
            -1,
            1000,
            100,
            GObject.ParamFlags.READWRITE,
        ),
        "repetition-penalty": (
            float,
            "Repetition Penalty",
            "Repetition penalty",
            1.0,
            2.0,
            1.1,
            GObject.ParamFlags.READWRITE,
        ),
        "max-model-len": (
            int,
            "Max Model Length",
            "Maximum model context length",
            512,
            65536,
            20480,
            GObject.ParamFlags.READWRITE,
        ),
        "trust-remote-code": (
            bool,
            "Trust Remote Code",
            "Trust remote code when loading model",
            True,
            GObject.ParamFlags.READWRITE,
        ),
        "gpu-memory-utilization": (
            float,
            "GPU Memory Utilization",
            "GPU memory fraction (0.0 to 1.0)",
            0.1,
            1.0,
            0.7,
            GObject.ParamFlags.READWRITE,
        ),
        "system-prompt": (
            str,
            "System Prompt",
            "System prompt for inference (optional)",
            None,
            GObject.ParamFlags.READWRITE,
        ),
        "queue-maxsize": (
            int,
            "Queue Maxsize",
            "Maximum size of inference queue",
            1,
            1000,
            20,
            GObject.ParamFlags.READWRITE,
        ),
        "max-wait-timeout": (
            int,
            "Max Wait Timeout",
            "Maximum wait time for segment completion (seconds)",
            1,
            3600,
            300,
            GObject.ParamFlags.READWRITE,
        ),
        "default-fps-numerator": (
            int,
            "Default FPS Numerator",
            "Default FPS numerator if not detected from stream",
            1,
            240,
            30,
            GObject.ParamFlags.READWRITE,
        ),
        "default-fps-denominator": (
            int,
            "Default FPS Denominator",
            "Default FPS denominator if not detected from stream",
            1,
            1000,
            1,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self) -> None:
        GstBase.BaseTransform.__init__(self)

        # Segment configuration
        self.segment_length_sec: int = DEFAULT_SEGMENT_LEN_SEC
        self.overlap_sec: int = DEFAULT_OVERLAP_SEC
        self.subsample_interval: int = DEFAULT_SUBSAMPLE_INTERVAL
        self.selection_fps: int = DEFAULT_SELECTION_FPS

        # Video format info
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.format: Optional[str] = None
        self.fps: Optional[tuple] = None  # (numerator, denominator)

        # Computed values
        self._step_ns: int = self._compute_step_ns()
        self._seg_len_ns: int = self.segment_length_sec * 1_000_000_000
        self._sample_interval_ns: Optional[int] = None

        # Model configuration (load from config)
        self.model: str = MODEL_PATH
        self.user_prompt: str = config.user_prompt
        self.max_tokens: int = config.max_tokens
        self.temperature: float = config.temperature
        self.gpu_id: int = config.gpu_id
        self.video_mode: int = config.video_mode
        self.tensor_format: str = config.tensor_format
        # Can be None if not specified
        self._system_prompt = config.system_prompt

        # Additional sampling parameters (optional)
        self.top_p: Optional[float] = config.top_p
        self.top_k: Optional[int] = config.top_k
        self.repetition_penalty: Optional[float] = config.repetition_penalty

        # Model initialization parameters
        self.max_model_len: int = config.max_model_len
        self.trust_remote_code: bool = config.trust_remote_code
        self.gpu_memory_utilization: float = config.gpu_memory_utilization

        # Pipeline parameters
        self.queue_maxsize: int = config.queue_maxsize
        self.max_wait_timeout: int = config.max_wait_timeout

        # Video parameters
        self.default_fps_numerator: int = config.default_fps[0]
        self.default_fps_denominator: int = config.default_fps[1]

        # Per-stream prompt overrides
        self._stream_prompts: Dict[int, Dict[str, Any]] = config.stream_prompts

        # Single VLM model instance (shared across all streams naturally)
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        # Per-stream contexts (keyed by pad_index/source_id)
        self.stream_contexts: Dict[int, StreamContext] = {}
        self.stream_contexts_lock: threading.Lock = threading.Lock()

        # Shared inference queue for all streams
        self._infer_queue: Queue = Queue(maxsize=self.queue_maxsize)
        self._infer_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

        # Load model once
        try:
            Gst.info(f"{GST_PLUGIN_NAME}: Loading VLM model '{self.model}'")

            # Initialize CUDA
            if torch.cuda.is_available():
                torch.cuda.init()

                # Determine which GPU to use
                if self.gpu_id >= 0:
                    # Explicit GPU ID specified
                    if self.gpu_id >= torch.cuda.device_count():
                        Gst.warning(
                            f"{GST_PLUGIN_NAME}: Requested GPU "
                            f"{self.gpu_id} not available. Using GPU 0 "
                            f"(available: {torch.cuda.device_count()} GPUs)"
                        )
                        self.gpu_id = 0
                    torch.cuda.set_device(self.gpu_id)
                    device_name = torch.cuda.get_device_name(self.gpu_id)
                    Gst.info(
                        f"{GST_PLUGIN_NAME}: CUDA initialized on GPU "
                        f"{self.gpu_id} ({device_name})"
                    )
                else:
                    # Auto-select from CUDA_VISIBLE_DEVICES
                    # (use device 0 of visible devices)
                    torch.cuda.set_device(0)
                    device_name = torch.cuda.get_device_name(0)
                    Gst.info(
                        f"{GST_PLUGIN_NAME}: CUDA initialized on GPU 0 "
                        f"(auto from CUDA_VISIBLE_DEVICES, {device_name})"
                    )
            else:
                Gst.error(f"{GST_PLUGIN_NAME}: CUDA not available!")
                raise RuntimeError("CUDA not available")

            self.llm = LLM(
                model=self.model,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model, trust_remote_code=self.trust_remote_code
                )
            except Exception:
                self.tokenizer = None

            Gst.info(f"{GST_PLUGIN_NAME}: VLM model loaded successfully")
        except Exception as e:
            Gst.error(f"{GST_PLUGIN_NAME}: Failed to initialize vLLM - {e}")
            import traceback

            traceback.print_exc()

    def _compute_step_ns(self) -> int:
        step = max(1, (self.segment_length_sec - self.overlap_sec))
        return step * 1_000_000_000

    def _update_sample_interval(self) -> None:
        if self.selection_fps and self.selection_fps > 0:
            self._sample_interval_ns = int(1_000_000_000 / self.selection_fps)
            Gst.info(
                f"{GST_PLUGIN_NAME}: selection_fps={self.selection_fps}, "
                f"sample_interval_ns={self._sample_interval_ns}"
            )
        else:
            self._sample_interval_ns = None
            Gst.info(
                f"{GST_PLUGIN_NAME}: selection_fps disabled, using "
                f"subsample-interval={self.subsample_interval}"
            )

    def do_get_property(self, prop: GObject.ParamSpec) -> Any:
        if prop.name == "model":
            return self.model
        if prop.name == "user-prompt":
            return self.user_prompt
        if prop.name == "max-tokens":
            return self.max_tokens
        if prop.name == "temperature":
            return self.temperature
        if prop.name == "gpu-id":
            return self.gpu_id
        if prop.name == "video-mode":
            return self.video_mode
        if prop.name == "tensor-format":
            return self.tensor_format
        if prop.name == "segment-length-sec":
            return self.segment_length_sec
        if prop.name == "overlap-sec":
            return self.overlap_sec
        if prop.name == "subsample-interval":
            return self.subsample_interval
        if prop.name == "selection-fps":
            return self.selection_fps
        if prop.name == "top-p":
            return self.top_p
        if prop.name == "top-k":
            return self.top_k
        if prop.name == "repetition-penalty":
            return self.repetition_penalty
        if prop.name == "max-model-len":
            return self.max_model_len
        if prop.name == "trust-remote-code":
            return self.trust_remote_code
        if prop.name == "gpu-memory-utilization":
            return self.gpu_memory_utilization
        if prop.name == "system-prompt":
            return self._system_prompt
        if prop.name == "queue-maxsize":
            return self.queue_maxsize
        if prop.name == "max-wait-timeout":
            return self.max_wait_timeout
        if prop.name == "default-fps-numerator":
            return self.default_fps_numerator
        if prop.name == "default-fps-denominator":
            return self.default_fps_denominator
        msg = f"{GST_PLUGIN_NAME}: Unknown property '{prop.name}'"
        raise AttributeError(msg)

    def do_set_property(self, prop: GObject.ParamSpec, value: Any) -> None:
        if prop.name == "model":
            self.model = value
        elif prop.name == "user-prompt":
            self.user_prompt = value
        elif prop.name == "max-tokens":
            self.max_tokens = value
        elif prop.name == "temperature":
            self.temperature = value
        elif prop.name == "gpu-id":
            self.gpu_id = int(value)
        elif prop.name == "video-mode":
            self.video_mode = int(value)
        elif prop.name == "tensor-format":
            self.tensor_format = str(value).lower()
            if self.tensor_format not in ["pytorch", "pil", "numpy"]:
                Gst.warning(
                    f"{GST_PLUGIN_NAME}: Invalid tensor-format '{value}', "
                    f"using 'pytorch'"
                )
                self.tensor_format = "pytorch"
        elif prop.name == "segment-length-sec":
            self.segment_length_sec = int(value)
            self._seg_len_ns = self.segment_length_sec * 1_000_000_000
            self._step_ns = self._compute_step_ns()
        elif prop.name == "overlap-sec":
            self.overlap_sec = int(value)
            self._step_ns = self._compute_step_ns()
        elif prop.name == "subsample-interval":
            self.subsample_interval = max(1, int(value))
        elif prop.name == "selection-fps":
            self.selection_fps = max(0, int(value))
            self._update_sample_interval()
        elif prop.name == "top-p":
            self.top_p = float(value) if value is not None else None
        elif prop.name == "top-k":
            self.top_k = int(value) if value is not None else None
        elif prop.name == "repetition-penalty":
            self.repetition_penalty = (  # noqa: BLK100
                float(value) if value is not None else None
            )
        elif prop.name == "max-model-len":
            self.max_model_len = int(value)
        elif prop.name == "trust-remote-code":
            self.trust_remote_code = bool(value)
        elif prop.name == "gpu-memory-utilization":
            self.gpu_memory_utilization = float(value)
        elif prop.name == "system-prompt":
            self._system_prompt = str(value) if value is not None else None
        elif prop.name == "queue-maxsize":
            self.queue_maxsize = int(value)
        elif prop.name == "max-wait-timeout":
            self.max_wait_timeout = int(value)
        elif prop.name == "default-fps-numerator":
            self.default_fps_numerator = int(value)
        elif prop.name == "default-fps-denominator":
            self.default_fps_denominator = int(value)
        else:
            msg = f"{GST_PLUGIN_NAME}: Unknown property '{prop.name}'"
            raise AttributeError(msg)

    def do_start(self) -> bool:
        if self.llm is None:
            Gst.error(f"{GST_PLUGIN_NAME}: vLLM not initialized")
            return False

        # Start inference worker thread
        self._stop_event.clear()
        self._infer_thread = threading.Thread(
            target=self._inference_worker, name="vlm-worker", daemon=True
        )
        self._infer_thread.start()

        self._update_sample_interval()
        msg = f"{GST_PLUGIN_NAME}: Plugin started - ready for multi-stream"
        Gst.info(msg)
        return True

    def do_set_caps(self, incaps: Gst.Caps, outcaps: Gst.Caps) -> bool:
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        self.format = struct.get_string("format")

        # get_fraction returns (success, numerator, denominator)
        fps_result = struct.get_fraction("framerate")
        if fps_result[0]:  # success
            self.fps = (
                fps_result[1],
                fps_result[2],
            )  # Store as tuple (numerator, denominator)
            Gst.info(
                f"{GST_PLUGIN_NAME}: caps set - {self.width}x{self.height} "
                f"format={self.format} fps={self.fps[0]}/{self.fps[1]}"
            )
        else:
            self.fps = (
                self.default_fps_numerator,
                self.default_fps_denominator,
            )
            Gst.info(
                f"{GST_PLUGIN_NAME}: caps set - {self.width}x{self.height} "
                f"format={self.format} fps={self.fps[0]}/{self.fps[1]} "
                f"(default)"
            )

        return True

    def _get_or_create_stream_context(self, stream_id: int) -> StreamContext:
        """Get or create context for a stream"""
        with self.stream_contexts_lock:
            if stream_id not in self.stream_contexts:
                ctx = StreamContext(stream_id)
                self.stream_contexts[stream_id] = ctx
                Gst.info(
                    f"{GST_PLUGIN_NAME}: Created context for stream "
                    f"{stream_id} (total streams: {len(self.stream_contexts)})"
                )
            return self.stream_contexts[stream_id]

    def _ensure_segments_until(
        self, ctx: StreamContext, pts_ns: int, batch_id: int
    ) -> None:
        """Create segments for a stream until covering pts_ns"""
        if ctx.next_segment_start_pts is None:
            ctx.next_segment_start_pts = pts_ns

        while (
            ctx.next_segment_start_pts is not None
            and ctx.next_segment_start_pts <= pts_ns
        ):
            start = ctx.next_segment_start_pts
            end = start + self._seg_len_ns
            seg = Segment(ctx.stream_id, start, end, batch_id)
            ctx.open_segments.append(seg)
            Gst.debug(
                f"{GST_PLUGIN_NAME}[Stream {ctx.stream_id}]: "
                f"opened segment [{start/1e9:.2f}s - {end/1e9:.2f}s]"
            )
            ctx.next_segment_start_pts = start + self._step_ns

    def _finalize_segments_up_to(
        self, ctx: StreamContext, pts_ns: int, batch_id: int
    ) -> None:
        """Finalize completed segments for a stream"""
        to_finalize = []
        for s in ctx.open_segments:
            batch_match = s.batch_id == batch_id or batch_id is None
            if s.end_pts_ns <= pts_ns and batch_match:
                to_finalize.append(s)

        # Determine if we're in multi-stream mode for cleaner logging
        num_streams = len(self.stream_contexts)
        stream_label = f"[Stream {ctx.stream_id}]" if num_streams > 1 else ""

        for seg in to_finalize:
            if seg.frames:
                start_sec = seg.start_pts_ns / 1_000_000_000
                end_sec = seg.end_pts_ns / 1_000_000_000
                print(
                    f"{GST_PLUGIN_NAME}{stream_label}: Finalizing segment "
                    f"[{start_sec:.2f}s - {end_sec:.2f}s] "
                    f"with {len(seg.frames)} frames"
                )

                # Build stream-specific prompt config with fallback to global
                prompt_config = {
                    "user_prompt": self._get_stream_config(
                        ctx.stream_id, "user_prompt", self.user_prompt
                    ),
                    "system_prompt": self._get_stream_config(
                        ctx.stream_id, "system_prompt", self._system_prompt
                    ),
                    "max_tokens": self._get_stream_config(
                        ctx.stream_id, "max_tokens", self.max_tokens
                    ),
                    "temperature": self._get_stream_config(
                        ctx.stream_id, "temperature", self.temperature
                    ),
                }

                # Add optional sampling parameters if specified
                top_p = self._get_stream_config(  # noqa: BLK100
                    ctx.stream_id, "top_p", self.top_p
                )
                if top_p is not None:
                    prompt_config["top_p"] = top_p

                top_k = self._get_stream_config(  # noqa: BLK100
                    ctx.stream_id, "top_k", self.top_k
                )
                if top_k is not None:
                    prompt_config["top_k"] = top_k

                repetition_penalty = self._get_stream_config(
                    ctx.stream_id,
                    "repetition_penalty",
                    self.repetition_penalty,
                )
                if repetition_penalty is not None:
                    prompt_config["repetition_penalty"] = repetition_penalty

                request = SegmentRequest(ctx.stream_id, seg, prompt_config)
                try:
                    self._infer_queue.put_nowait(request)
                    ctx.segments_submitted += 1
                    ctx.total_frames_in_segments += len(seg.frames)
                    print(
                        f"{GST_PLUGIN_NAME}{stream_label}: Submitted "
                        f"(total: {ctx.segments_submitted})"
                    )
                except Exception as e:
                    ctx.segments_dropped += 1
                    print(f"{GST_PLUGIN_NAME}{stream_label}: Dropped - {e}")

            try:
                ctx.open_segments.remove(seg)
            except ValueError:
                pass

    def do_transform_ip(self, gst_buffer: Gst.Buffer) -> Gst.FlowReturn:
        """Process batched frames from multiple streams"""

        buffer = Buffer(gst_buffer)
        batch_meta = buffer.batch_meta

        if batch_meta.n_frames == 0:
            return Gst.FlowReturn.OK

        # Process each frame in the batch
        for frame_meta in batch_meta.frame_items:
            try:
                # Identify which stream this frame belongs to
                stream_id = frame_meta.pad_index  # or frame_meta.source_id

                # Get or create context for this stream
                ctx = self._get_or_create_stream_context(stream_id)

                # Process frame for this stream's context
                ctx.frame_counter += 1
                subsample_mod = ctx.frame_counter % self.subsample_interval
                keep_by_subsample = subsample_mod == 0

                current_pts = frame_meta.buffer_pts

                # Ensure segments and finalize completed ones
                batch_id = frame_meta.batch_id
                self._ensure_segments_until(ctx, current_pts, batch_id)
                prev_pts = current_pts - 1
                self._finalize_segments_up_to(ctx, prev_pts, batch_id)

                # Extract frame data
                tensor = buffer.extract(batch_id)
                torch_frame_data = torch.utils.dlpack.from_dlpack(tensor)

                if torch_frame_data is None or torch_frame_data.numel() == 0:
                    continue

                bd_created = False
                bd = None

                # Add frame to appropriate segments for this stream
                if (
                    self._sample_interval_ns is not None
                    and self._sample_interval_ns > 0
                ):
                    # FPS-based sampling
                    for seg in ctx.open_segments:
                        if seg.batch_id != frame_meta.batch_id:
                            continue
                        if seg.start_pts_ns <= current_pts <= seg.end_pts_ns:
                            should_keep = (
                                seg.last_saved_pts_ns is None
                                or (current_pts - seg.last_saved_pts_ns)
                                >= self._sample_interval_ns
                            )
                            if should_keep:
                                if not bd_created:
                                    bd = BufferData(
                                        -1,
                                        current_pts,
                                        -1,
                                        -1,
                                        torch_frame_data.clone(),
                                    )
                                    bd_created = True
                                seg.frames.append(bd)
                                seg.last_saved_pts_ns = current_pts
                else:
                    # Interval-based sampling
                    if keep_by_subsample:
                        bd = BufferData(
                            -1, current_pts, -1, -1, torch_frame_data.clone()
                        )
                        for seg in ctx.open_segments:
                            start = seg.start_pts_ns
                            end = seg.end_pts_ns
                            if start <= current_pts <= end:
                                seg.frames.append(bd)
                                seg.last_saved_pts_ns = current_pts

            except Exception as e:
                msg = f"{GST_PLUGIN_NAME}: Frame processing failed - {e}"
                Gst.warning(msg)

        return Gst.FlowReturn.OK

    def _get_stream_config(  # noqa: BLK100
        self, stream_id: int, setting: str, default: Any
    ) -> Any:
        """
        Get config value for a specific stream with fallback to global

        Priority:
        1. Stream-specific setting (if exists in stream_prompts)
        2. Global setting (self.setting)
        3. Default value

        Args:
            stream_id: Stream identifier
            setting: Setting name (e.g., 'user_prompt', 'system_prompt')
            default: Default value if not found

        Returns:
            Config value for this stream
        """
        # Check if stream has specific override
        if (
            stream_id in self._stream_prompts
            and setting in self._stream_prompts[stream_id]
        ):
            value = self._stream_prompts[stream_id][setting]
            Gst.debug(
                f"{GST_PLUGIN_NAME}[Stream {stream_id}]: "
                f"Using stream-specific {setting}: {value}"
            )
            return value

        # Fall back to global setting or default
        return default

    def _format_user_prompt(
        self,
        user_prompt: str,
        stream_id: int,
        num_frames: int,
        timestamps: str,
    ) -> str:
        """
        Format user prompt by replacing placeholders.

        Available placeholders:
        - {num_frames}: Number of frames in segment
        - {stream_id}: Stream identifier
        - {timestamps}: Timestamp string (e.g., "0.00s 1.00s 2.00s")

        Args:
            user_prompt: User's prompt string with placeholders
            stream_id: Stream ID
            num_frames: Number of frames
            timestamps: Timestamp string

        Returns:
            Formatted prompt string
        """
        try:
            return user_prompt.format(
                num_frames=num_frames,
                stream_id=stream_id,
                timestamps=timestamps,
            )
        except KeyError as e:
            msg = f"{GST_PLUGIN_NAME}: Invalid placeholder in user_prompt: {e}"
            Gst.warning(msg)
            return user_prompt

    def _convert_tensor_to_format(
        self, tensor: torch.Tensor, target_format: str
    ):  # noqa: BLK100
        """
        Convert PyTorch tensor to specified format (pytorch, pil, or numpy)

        Args:
            tensor: PyTorch tensor with shape [C, H, W], RGB format
            target_format: "pytorch", "pil", or "numpy"

        Returns:
            Converted tensor in requested format
        """
        if target_format == "pytorch":
            return tensor.cpu()

        elif target_format == "pil":
            # Convert [C, H, W] to [H, W, C] for PIL
            tensor = tensor.cpu()

            # Handle different dtypes
            if tensor.dtype in (torch.float32, torch.float16):
                # Assume normalized [0, 1], convert to [0, 255] uint8
                tensor = (tensor * 255).clamp(0, 255).byte()
            elif tensor.dtype == torch.uint8:
                # Already uint8
                pass
            else:
                # Convert to uint8
                tensor = tensor.byte()

            # Convert to numpy and create PIL Image
            # [C, H, W] -> [H, W, C]
            np_array = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(np_array, mode="RGB")

        elif target_format == "numpy":
            # Convert to numpy, keep shape [C, H, W]
            return tensor.cpu().numpy()

        else:
            Gst.warning(
                f"{GST_PLUGIN_NAME}: Unknown tensor format "
                f"'{target_format}', using pytorch"
            )
            return tensor.cpu()

    def _inference_worker(self) -> None:
        """Worker thread processes segments from all streams"""
        print(f"{GST_PLUGIN_NAME}: Inference worker started (multi-stream)")

        while not self._stop_event.is_set():
            try:
                request = self._infer_queue.get(timeout=0.1)
            except Empty:
                continue

            stream_id = request.stream_id
            segment = request.segment
            start_sec = segment.start_pts_ns / 1_000_000_000
            end_sec = segment.end_pts_ns / 1_000_000_000

            # Determine stream label based on number of active streams
            num_streams = len(self.stream_contexts)
            stream_label = f"[Stream {stream_id}]" if num_streams > 1 else ""

            msg = (
                f"{GST_PLUGIN_NAME}{stream_label}: Processing segment "
                f"[{start_sec:.2f}s - {end_sec:.2f}s] "
                f"with {len(segment.frames)} frames"
            )
            print(msg)

            try:
                result_text = self._run_vlm_batch(
                    segment, request.prompt_config
                )  # noqa: BLK100
                if result_text:
                    ctx = self.stream_contexts.get(stream_id)
                    if ctx:
                        ctx.update_result(result_text, start_sec, end_sec)
                        print(
                            f"{GST_PLUGIN_NAME}{stream_label}: Completed "
                            f"(total: {ctx.segments_completed})"
                        )
                        # Print full result with timestamp prefix
                        print(
                            f"{GST_PLUGIN_NAME}{stream_label}: Result: "
                            f"{start_sec:.2f}s-{end_sec:.2f}s {result_text}"
                        )

                        # Emit signal with result
                        self.emit(
                            "vlm-result",
                            stream_id,
                            start_sec,
                            end_sec,
                            result_text,
                        )
                else:
                    print(
                        f"{GST_PLUGIN_NAME}{stream_label}: "
                        f"VLM returned empty result"
                    )
            except Exception as e:
                msg = f"{GST_PLUGIN_NAME}{stream_label}: Worker error - {e}"
                print(msg)
                import traceback

                traceback.print_exc()
            finally:
                self._infer_queue.task_done()

        print(f"{GST_PLUGIN_NAME}: Inference worker stopped")

    def _run_vlm_batch(
        self, segment: Segment, prompt_config: Dict
    ) -> Optional[str]:  # noqa: BLK100
        """Run VLM inference on a segment"""
        if self.llm is None:
            return None

        try:
            # Collect frame tensors and timestamps
            frame_tensors = []
            frame_times = []

            for i, frame_data in enumerate(segment.frames):
                tensor = frame_data.tensor_gpu
                if tensor.dim() == 3:
                    tensor = tensor.permute(2, 0, 1)
                frame_tensors.append(tensor)
                frame_time_sec = frame_data.pts / 1_000_000_000.0
                frame_times.append(frame_time_sec)

            if not frame_tensors:
                return None

            # Stack into batch
            batch_tensor = torch.stack(frame_tensors)

            # Calculate FPS
            if len(frame_times) > 1:
                time_diff = frame_times[-1] - frame_times[0]
                num_intervals = len(frame_times) - 1
                fps = num_intervals / time_diff if time_diff > 0 else 1.0
            else:
                fps = 1.0

            # Video metadata
            if len(frame_times) > 1:
                duration = frame_times[-1] - frame_times[0]
            else:
                duration = 0.0
            video_metadata = {
                "total_num_frames": len(frame_tensors),
                "frames_indices": [int(t * fps) for t in frame_times],
                "fps": fps,
                "duration": duration,
            }

            # Build timestamp string
            string_of_times = " ".join([f"{t:.2f}s" for t in frame_times])
            num_frames = len(frame_tensors)

            # Build SamplingParams with required parameters
            sampling_params_dict = {
                "temperature": prompt_config.get("temperature", 0.2),
                "max_tokens": prompt_config.get("max_tokens", 64),
            }

            # Add optional parameters if specified
            if "top_p" in prompt_config and prompt_config["top_p"] is not None:
                sampling_params_dict["top_p"] = prompt_config["top_p"]

            if "top_k" in prompt_config and prompt_config["top_k"] is not None:
                sampling_params_dict["top_k"] = prompt_config["top_k"]

            if (
                "repetition_penalty" in prompt_config
                and prompt_config["repetition_penalty"] is not None
            ):
                sampling_params_dict["repetition_penalty"] = prompt_config[
                    "repetition_penalty"
                ]

            sampling_params = SamplingParams(**sampling_params_dict)

            # Use chat template
            has_chat_template = hasattr(self.tokenizer, "apply_chat_template")
            if self.tokenizer and has_chat_template:
                # Determine mode: video_mode=1 uses video input (all frames),
                # video_mode=0 uses multi-image input
                if self.video_mode == 0:
                    # Image mode: pass all frames as separate images
                    # Get user prompt with default
                    user_prompt = prompt_config.get(
                        "user_prompt",
                        "These are {num_frames} images from stream "
                        "{stream_id} sampled at timestamps {timestamps}. "
                        "Describe the scene in detail.",
                    )

                    # Format prompt with placeholders
                    prompt_text = self._format_user_prompt(
                        user_prompt,
                        segment.stream_id,
                        num_frames,
                        string_of_times,
                    )

                    # Build content with text + multiple images
                    content = [{"type": "text", "text": prompt_text}]
                    for i in range(num_frames):
                        img_entry = {"type": "image", "image": f"frame{i}.jpg"}
                        content.append(img_entry)

                    # Build messages - only include system prompt if not None
                    system_prompt = prompt_config.get(
                        "system_prompt", self._system_prompt
                    )
                    if system_prompt is not None:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content},
                        ]
                    else:
                        messages = [
                            {"role": "user", "content": content},
                        ]

                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Convert frames to target format (pytorch/pil/numpy)
                    image_tensors = [
                        self._convert_tensor_to_format(
                            batch_tensor[i], self.tensor_format
                        )
                        for i in range(num_frames)
                    ]
                    inputs = {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image_tensors},
                    }

                elif num_frames == 1:
                    # Single frame in video_mode=1: use image input.
                    # Video processors (e.g. Qwen3VL) require >=2 frames;
                    # passing as image avoids that constraint.
                    user_prompt = prompt_config.get(
                        "user_prompt",
                        "This is an image from stream {stream_id} at "
                        "timestamp {timestamps}. Describe the scene.",
                    )

                    prompt_text = self._format_user_prompt(
                        user_prompt,
                        segment.stream_id,
                        num_frames,
                        string_of_times,
                    )

                    system_prompt = prompt_config.get(
                        "system_prompt", self._system_prompt
                    )
                    if system_prompt is not None:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image", "image": "frame.jpg"},
                                ],
                            },
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image", "image": "frame.jpg"},
                                ],
                            },
                        ]

                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Convert [C, H, W] tensor to [H, W, C] numpy array.
                    # vLLM interprets a torch.Tensor in image mm_data as
                    # pre-computed image_embeds; numpy avoids that error.
                    image_np = (
                        batch_tensor[0].cpu().permute(1, 2, 0).numpy()
                    )
                    inputs = {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image_np},
                    }

                else:
                    # Video mode - multiple frames with video metadata
                    # Get user prompt with default
                    user_prompt = prompt_config.get(
                        "user_prompt",
                        "This is a video from stream {stream_id} with "
                        "{num_frames} frames sampled at timestamps "
                        "{timestamps}. Describe the video content.",
                    )

                    # Format prompt with placeholders
                    prompt_text = self._format_user_prompt(
                        user_prompt,
                        segment.stream_id,
                        num_frames,
                        string_of_times,
                    )

                    # Build messages - only include system prompt if not None
                    system_prompt = prompt_config.get(
                        "system_prompt", self._system_prompt
                    )
                    if system_prompt is not None:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "video", "video": "segment.mp4"},
                                ],
                            },
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "video", "video": "segment.mp4"},
                                ],
                            },
                        ]

                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    video_input = (batch_tensor.cpu(), video_metadata)
                    inputs = {
                        "prompt": prompt_text,
                        "multi_modal_data": {"video": video_input},
                    }

                outputs = self.llm.generate(
                    inputs, sampling_params=sampling_params
                )  # noqa: BLK100
                if outputs:
                    return outputs[0].outputs[0].text

            return None

        except Exception as e:
            msg = f"{GST_PLUGIN_NAME}: VLM inference failed - {e}"
            print(msg)
            import traceback

            traceback.print_exc()
            return None

    def do_stop(self) -> bool:
        """Stop processing and finalize all streams"""
        n = len(self.stream_contexts)
        lbl = "stream" if n == 1 else "stream(s)"
        print(f"{GST_PLUGIN_NAME}: Stopping - finalizing {n} {lbl}")

        # Finalize remaining segments for each stream
        with self.stream_contexts_lock:
            for stream_id, ctx in self.stream_contexts.items():
                ctx_label = f"[Stream {stream_id}]" if n > 1 else ""

                if ctx.open_segments:
                    print(
                        f"{GST_PLUGIN_NAME}{ctx_label}: Finalizing "
                        f"{len(ctx.open_segments)} remaining segments"
                    )
                    for seg in ctx.open_segments[:]:
                        if seg.frames:
                            start_sec = seg.start_pts_ns / 1_000_000_000
                            end_sec = seg.end_pts_ns / 1_000_000_000
                            pts_diff = seg.frames[-1].pts - seg.frames[0].pts
                            if len(seg.frames) > 1:
                                duration_sec = pts_diff / 1_000_000_000
                            else:
                                duration_sec = 0

                            msg = (
                                f"{GST_PLUGIN_NAME}{ctx_label}: Submitting "
                                f"incomplete segment "
                                f"[{start_sec:.2f}s - {end_sec:.2f}s] "
                                f"with {len(seg.frames)} frames "
                                f"(duration: {duration_sec:.2f}s)"
                            )
                            print(msg)

                            prompt_config = {
                                "user_prompt": self.user_prompt,
                                "system_prompt": self._system_prompt,
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature,
                            }

                            # Add optional sampling parameters if specified
                            if self.top_p is not None:
                                prompt_config["top_p"] = self.top_p
                            if self.top_k is not None:
                                prompt_config["top_k"] = self.top_k
                            if self.repetition_penalty is not None:
                                prompt_config["repetition_penalty"] = (
                                    self.repetition_penalty
                                )

                            request = SegmentRequest(
                                stream_id, seg, prompt_config
                            )  # noqa: BLK100
                            try:
                                self._infer_queue.put_nowait(request)
                                ctx.segments_submitted += 1
                                ctx.total_frames_in_segments += len(seg.frames)
                            except Exception:
                                ctx.segments_dropped += 1

        # Wait for queue to drain AND all processing to complete
        import time

        max_wait = self.max_wait_timeout
        elapsed = 0.0

        print(f"{GST_PLUGIN_NAME}: Waiting for all segments to complete...")

        while elapsed < max_wait:
            # Check both queue and per-stream completion status
            queue_empty = self._infer_queue.empty()

            all_complete = True
            pending_count = 0
            with self.stream_contexts_lock:
                for ctx in self.stream_contexts.values():
                    pending = (
                        ctx.segments_submitted
                        - ctx.segments_completed
                        - ctx.segments_dropped
                    )
                    if pending > 0:
                        all_complete = False
                        pending_count += pending

            if queue_empty and all_complete:
                print(
                    f"{GST_PLUGIN_NAME}: All segments completed "
                    f"successfully (waited {elapsed:.1f}s)"
                )
                break

            time.sleep(0.5)
            elapsed += 0.5

            if int(elapsed) % 10 == 0 and pending_count > 0:
                print(
                    f"{GST_PLUGIN_NAME}: Still processing... "
                    f"{pending_count} segment(s) remaining "
                    f"({elapsed:.0f}s elapsed)"
                )

        if not all_complete:
            print(
                f"{GST_PLUGIN_NAME}: WARNING: Timeout waiting for "
                f"segments. Some segments may not be fully processed."
            )

        # Give a small grace period to ensure any final result updates
        # are complete
        time.sleep(0.5)

        # Stop worker thread
        print(f"\n{GST_PLUGIN_NAME}: Stopping inference worker...")
        self._stop_event.set()
        if self._infer_thread:
            self._infer_thread.join(timeout=5.0)
            self._infer_thread = None

        # Shut down vLLM engine to release GPU memory
        if self.llm is not None:
            print(f"{GST_PLUGIN_NAME}: Shutting down vLLM engine...")
            try:
                if hasattr(self.llm, "shutdown"):
                    self.llm.shutdown()
                elif hasattr(self.llm, "llm_engine") and hasattr(
                    self.llm.llm_engine, "shutdown"
                ):
                    self.llm.llm_engine.shutdown()
            except Exception as e:
                print(
                    f"{GST_PLUGIN_NAME}: Warning: vLLM shutdown error: {e}"
                )
            finally:
                self.llm = None

        # Print statistics for each stream
        # (AFTER worker stops to ensure final counts)
        num_streams = len(self.stream_contexts)
        if num_streams == 1:
            stats_label = "Statistics"
        else:
            stats_label = "Multi-Stream Statistics"
        print(f"\n{GST_PLUGIN_NAME}: Final {stats_label}:")

        with self.stream_contexts_lock:
            for stream_id, ctx in self.stream_contexts.items():
                # For single stream, omit "Stream 0:" for cleaner output
                if num_streams == 1:
                    print(f"  Frames processed: {ctx.frame_counter}")
                    print(f"  Segments submitted: {ctx.segments_submitted}")
                    print(f"  Segments completed: {ctx.segments_completed}")
                    print(f"  Segments dropped: {ctx.segments_dropped}")
                    total_frames = ctx.total_frames_in_segments
                    print(f"  Total frames in segments: {total_frames}")
                    if ctx.segments_submitted > 0:
                        n_frames = ctx.total_frames_in_segments
                        n_segs = ctx.segments_submitted
                        avg_frames = n_frames / n_segs
                        print(f"  Avg frames per segment: {avg_frames:.1f}")
                else:
                    print(f"\n  Stream {stream_id}:")
                    print(f"    Frames processed: {ctx.frame_counter}")
                    print(f"    Segments submitted: {ctx.segments_submitted}")
                    print(f"    Segments completed: {ctx.segments_completed}")
                    print(f"    Segments dropped: {ctx.segments_dropped}")
                    total_frames = ctx.total_frames_in_segments
                    print(f"    Total frames in segments: {total_frames}")
                    if ctx.segments_submitted > 0:
                        n_frames = ctx.total_frames_in_segments
                        n_segs = ctx.segments_submitted
                        avg_frames = n_frames / n_segs
                        print(f"    Avg frames per segment: {avg_frames:.1f}")

        print(f"{GST_PLUGIN_NAME}: Shutdown complete")
        return True


GObject.type_register(NvVllmVLM)
__gstelementfactory__ = (GST_PLUGIN_NAME, Gst.Rank.NONE, NvVllmVLM)
