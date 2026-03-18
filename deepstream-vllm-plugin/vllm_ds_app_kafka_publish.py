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

"""
DeepStream VLM application with Kafka publishing and signal-based result
handling.
Supports both single-stream and multi-stream processing with file and RTSP
sources.

Features:
- Single-stream and multi-stream VLM processing
- Real-time result delivery via GObject signals
- Kafka topic publishing for downstream processing
- Dry-run mode for testing without Kafka
- Efficient event-driven architecture
- File and RTSP source support via uridecodebin
"""

import json
import os
import sys
import time
from typing import Optional

import gi
gi.require_version("Gst", "1.0")  # noqa: E402, I003, BLK100
from gi.repository import GLib, Gst  # noqa: E402, I003

# Register the custom plugin
import gstnvvllmvlm  # noqa: E402

Gst.Element.register(None, "nvvllmvlm", Gst.Rank.NONE, gstnvvllmvlm.NvVllmVLM)

# Kafka imports (with graceful fallback)
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Run: pip install kafka-python")


def to_uri(path_or_uri: str) -> str:
    """Convert a file path or URI string to a GStreamer-compatible URI."""
    if "://" in path_or_uri:
        return path_or_uri
    return "file://" + os.path.abspath(path_or_uri)


class VLMKafkaSignalPublisher:
    """
    Kafka publisher that uses GObject signals to receive VLM results.
    More efficient than polling - publishes immediately when results are
    available.
    """

    def __init__(self, kafka_config: dict, topic: str, dry_run: bool = False):
        """
        Initialize Kafka publisher.

        Args:
            kafka_config: Kafka connection configuration
            topic: Topic name to publish to
            dry_run: If True, print messages instead of sending to Kafka
        """
        self.topic = topic
        self.dry_run = dry_run
        self.producer: Optional[KafkaProducer] = None
        self.messages_sent = 0
        self.messages_failed = 0

        # Initialize Kafka producer
        if not dry_run and KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=kafka_config.get(
                        "bootstrap_servers", "localhost:9092"
                    ),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    # Required for idempotence
                    max_in_flight_requests_per_connection=1,
                    enable_idempotence=True,
                    compression_type="gzip",
                    linger_ms=100,
                    batch_size=16384,
                )
                print(f"✓ Kafka producer initialized (topic: {self.topic})")
            except Exception as e:
                print(f"✗ Failed to initialize Kafka producer: {e}")
                print("  Falling back to dry-run mode (console output only)")
                self.dry_run = True
                self.producer = None
                print("✓ Dry-run mode enabled")
        else:
            if not KAFKA_AVAILABLE:
                print("✗ Kafka not available - dry-run mode enabled")
            else:
                print("✓ Dry-run mode enabled (console output only)")
            self.producer = None

    def on_vlm_result(
        self, element, stream_id, start_time, end_time, result_text
    ):
        """
        Signal handler for vlm-result signal.
        Called immediately when VLM inference completes.

        Args:
            element: The nvvllmvlm element that emitted the signal
            stream_id: Stream identifier
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            result_text: VLM inference result
        """
        # Construct message
        message = {
            "stream_id": stream_id,
            "timestamp": time.time(),
            "segment": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            },
            "result": result_text,
            "metadata": {"source": "vllm-ds-plugin", "version": "1.0"},
        }

        # Publish to Kafka or print to console
        self.publish(message, stream_id)

    def publish(self, message: dict, stream_id: int):
        """
        Publish message to Kafka or print to console.

        Args:
            message: Message payload
            stream_id: Stream ID (used as partition key)
        """
        # Use stream_id as partition key for ordering
        partition_key = f"stream_{stream_id}"

        if self.dry_run or self.producer is None:
            # Dry-run mode: print to console
            print(f"\n{'='*80}")
            print("📤 KAFKA MESSAGE (Dry-Run)")
            print(f"{'='*80}")
            print(f"Topic: {self.topic}")
            print(f"Key: {partition_key}")
            print(f"Value: {json.dumps(message, indent=2)}")
            print(f"{'='*80}\n")
            self.messages_sent += 1
        else:
            # Send to Kafka
            try:
                future = self.producer.send(
                    self.topic, key=partition_key, value=message
                )

                # Optional: wait for acknowledgment
                record_metadata = future.get(timeout=10)

                self.messages_sent += 1
                print(
                    f"✓ Published to Kafka: stream={stream_id}, "
                    f"time={message['segment']['start_time']:.1f}s-"
                    f"{message['segment']['end_time']:.1f}s, "
                    f"partition={record_metadata.partition}, "
                    f"offset={record_metadata.offset}"
                )

            except KafkaError as e:
                self.messages_failed += 1
                print(f"✗ Kafka publish failed: {e}")
            except Exception as e:
                self.messages_failed += 1
                print(f"✗ Unexpected error during publish: {e}")

    def close(self):
        """Close Kafka producer and print statistics"""
        if self.producer:
            print("\nFlushing Kafka producer...")
            self.producer.flush(timeout=10)
            self.producer.close()

        print(f"\n{'='*80}")
        print("KAFKA PUBLISHER STATISTICS")
        print(f"{'='*80}")
        print(f"Messages sent: {self.messages_sent}")
        print(f"Messages failed: {self.messages_failed}")
        print(f"{'='*80}\n")


class VLMKafkaApp:
    """DeepStream VLM app with Kafka publishing via signals
    (single or multi-stream, file or RTSP sources)"""

    def __init__(self, input_uris, kafka_config, topic, dry_run=False):
        """
        Initialize application.

        Args:
            input_uris: List of GStreamer-compatible URIs (file:// or rtsp://)
            kafka_config: Kafka connection configuration
            topic: Kafka topic name
            dry_run: If True, print messages instead of sending to Kafka
        """
        self.input_uris = input_uris
        self.num_sources = len(input_uris)
        self.pipeline = None
        self.loop = None
        self.streams_eos = set()

        # Initialize Kafka publisher
        self.kafka_publisher = VLMKafkaSignalPublisher(
            kafka_config, topic, dry_run
        )

    def bus_call(self, bus, message, loop):
        """Handle GStreamer bus messages"""
        t = message.type

        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()

        return True

    def pad_probe_callback(self, pad, info, stream_id):
        """Probe to detect per-stream EOS"""
        gst_buffer = info.get_buffer()
        if gst_buffer:
            if gst_buffer.pts == Gst.CLOCK_TIME_NONE:
                print(f"Stream {stream_id}: Received EOS")
                self.streams_eos.add(stream_id)

                if len(self.streams_eos) == self.num_sources:
                    print(f"All {self.num_sources} stream(s) finished")

        return Gst.PadProbeReturn.OK

    def build_pipeline(self):
        """Build the GStreamer pipeline.

        Uses uridecodebin per stream so that both file:// and rtsp:// URIs
        are supported transparently. uridecodebin selects the appropriate
        source plugin, demuxer, parser, and hardware decoder automatically.
        """
        print(f"Building pipeline for {self.num_sources} source(s)...")

        has_live = any(
            uri.startswith("rtsp://") or uri.startswith("rtsps://")
            for uri in self.input_uris
        )

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("vlm-kafka-signal-pipeline")

        # Create streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", self.num_sources)
        streammux.set_property("live-source", has_live)
        if not has_live:
            streammux.set_property("batched-push-timeout", 4000000)

        # Add to pipeline
        self.pipeline.add(streammux)

        # Pre-request mux sink pads so pad-added callbacks can link into them
        mux_sink_pads = []
        for i in range(self.num_sources):
            sink_pad = streammux.request_pad_simple(f"sink_{i}")
            if not sink_pad:
                print(f"Error: Could not get sink pad {i} from streammux")
                return None
            sink_pad.add_probe(
                Gst.PadProbeType.BUFFER, self.pad_probe_callback, i
            )
            mux_sink_pads.append(sink_pad)

        # Create one uridecodebin per source
        for i, uri in enumerate(self.input_uris):
            print(f"  Source {i}: {uri}")

            uri_decode_bin = Gst.ElementFactory.make(
                "uridecodebin", f"uri-decode-bin-{i}"
            )
            if not uri_decode_bin:
                print(
                    f"Error: Could not create uridecodebin for stream {i}"
                )
                return None

            uri_decode_bin.set_property("uri", uri)
            self.pipeline.add(uri_decode_bin)

            # Capture loop variables via default args
            def on_pad_added(
                element,
                pad,
                mux_sinkpad=mux_sink_pads[i],
                stream_id=i,
            ):
                caps = pad.get_current_caps()
                if not caps:
                    caps = pad.query_caps()
                if not caps:
                    return
                structure = caps.get_structure(0)
                if "video" in structure.get_name():
                    if not mux_sinkpad.is_linked():
                        if pad.link(mux_sinkpad) == Gst.PadLinkReturn.OK:
                            print(
                                f"  Linked uridecodebin → "
                                f"streammux.sink_{stream_id}"
                            )

            uri_decode_bin.connect("pad-added", on_pad_added)

        # Video converter
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvvidconv.set_property("nvbuf-memory-type", 0)

        # Caps filter for RGB
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        caps_rgb = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGB")
        caps_filter.set_property("caps", caps_rgb)

        # VLM plugin - uses configuration from config.yaml
        nvvllm = Gst.ElementFactory.make("nvvllmvlm", "vlm-infer")

        # Connect signal to Kafka publisher
        nvvllm.connect("vlm-result", self.kafka_publisher.on_vlm_result)
        print("✓ Connected vlm-result signal to Kafka publisher")

        # Fakesink
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")
        sink.set_property("sync", False)

        # Add elements to pipeline
        self.pipeline.add(nvvidconv)
        self.pipeline.add(caps_filter)
        self.pipeline.add(nvvllm)
        self.pipeline.add(sink)

        # Link pipeline
        streammux.link(nvvidconv)
        nvvidconv.link(caps_filter)
        caps_filter.link(nvvllm)
        nvvllm.link(sink)

        print("Pipeline built successfully\n")

        return self.pipeline

    def run(self):
        """Run the application"""
        # Build pipeline
        pipeline = self.build_pipeline()

        # Set up bus
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        # Create main loop
        self.loop = GLib.MainLoop()
        bus.connect("message", self.bus_call, self.loop)

        # Start pipeline
        print("Starting pipeline...")
        pipeline.set_state(Gst.State.PLAYING)

        try:
            print("Running... (Press Ctrl+C to stop)\n")
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")

        # Cleanup
        print("\nStopping pipeline...")
        pipeline.set_state(Gst.State.NULL)

        # Close Kafka publisher
        self.kafka_publisher.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepStream VLM app with Kafka publishing "
                    "(single-stream or multi-stream, file or RTSP sources)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
URIs can be:
  File paths:   /path/to/video.mp4  (auto-converted to file:// URI)
  File URIs:    file:///path/to/video.mp4
  RTSP streams: rtsp://user:pass@host:port/stream

Examples:
  # Single file with dry-run (console output)
  python3 vllm_ds_app_kafka_publish.py video1.mp4 --dry-run

  # RTSP stream with dry-run
  python3 vllm_ds_app_kafka_publish.py rtsp://192.168.1.100:8554/stream \\
      --dry-run

  # Single file with Kafka publishing
  python3 vllm_ds_app_kafka_publish.py video1.mp4 \\
      --kafka-bootstrap localhost:9092 \\
      --topic vlm-results

  # Multi-stream with mixed sources and Kafka
  python3 vllm_ds_app_kafka_publish.py \\
      video1.mp4 rtsp://192.168.1.100:8554/stream \\
      --kafka-bootstrap localhost:9092 \\
      --topic vlm-results
        """,
    )

    parser.add_argument(
        "sources",
        nargs="+",
        help="Video file paths or URIs to process (file paths, file://, rtsp://)",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--topic",
        default="vlm-results",
        help="Kafka topic name (default: vlm-results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages to console instead of sending to Kafka",
    )

    args = parser.parse_args()

    # Initialize GStreamer
    Gst.init(None)

    # Convert bare file paths to file:// URIs; validate files exist
    input_uris = []
    for src in args.sources:
        uri = to_uri(src)
        if uri.startswith("file://"):
            file_path = uri[len("file://"):]
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
        input_uris.append(uri)

    # Kafka configuration
    kafka_config = {"bootstrap_servers": args.kafka_bootstrap}

    # Create and run app
    app = VLMKafkaApp(
        input_uris=input_uris,
        kafka_config=kafka_config,
        topic=args.topic,
        dry_run=args.dry_run,
    )
    app.run()


if __name__ == "__main__":
    main()
