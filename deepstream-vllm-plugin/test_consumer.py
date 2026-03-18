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
Kafka consumer for VLM results.
Continuously listens for VLM inference results and prints them in real-time.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime

try:
    from kafka import KafkaConsumer
except ImportError:
    print("Error: kafka-python not installed")
    print("Install with: pip install kafka-python")
    sys.exit(1)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Kafka consumer for VLM results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Listen for messages with 5-minute timeout (default behavior)
  python3 test_consumer.py

  # Specify custom Kafka broker and topic
  python3 test_consumer.py --broker localhost:9092 --topic vlm-result

  # Read all existing messages from beginning with a fresh consumer group
  python3 test_consumer.py --reset --topic vlm-result

  # Keep running forever until Ctrl+C (no timeout)
  python3 test_consumer.py --timeout 0

  # Exit after 30 seconds of no new messages
  python3 test_consumer.py --timeout 30000

  # Start from latest (only read new messages going forward)
  python3 test_consumer.py --from-latest
        """,
    )
    parser.add_argument(
        "--broker",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--topic",
        default="vlm-results",
        help="Kafka topic name (default: vlm-results)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Use unique consumer group to read all messages from beginning",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300000,
        help=(
            "Exit after N milliseconds of no messages "
            "(default: 300000 = 5 minutes, use 0 for no timeout)"
        ),
    )
    parser.add_argument(
        "--from-latest",
        action="store_true",
        help="Start from latest messages instead of earliest",
    )

    args = parser.parse_args()

    bootstrap_servers = args.broker
    topic = args.topic

    # Consumer group
    if args.reset:
        group_id = f"vlm-consumer-{uuid.uuid4().hex[:8]}"
        print(
            "🔄 RESET MODE: Using unique consumer group "  # noqa: BLK100
            "(will read from beginning)"
        )
    else:
        group_id = "vlm-consumer-group"

    # Offset reset
    offset_reset = "latest" if args.from_latest else "earliest"

    # Handle timeout (0 means no timeout)
    timeout_ms = None if args.timeout == 0 else args.timeout

    print("Starting Kafka consumer...")
    print(f"  Bootstrap servers: {bootstrap_servers}")
    print(f"  Topic: {topic}")
    print(f"  Consumer group: {group_id}")
    print(f"  Reading from: {offset_reset}")
    if timeout_ms:
        timeout_mins = timeout_ms / 60000
        print(
            f"  Timeout: {timeout_ms}ms ({timeout_mins:.1f} minutes - "
            "will exit if no messages)"
        )
    else:
        print("  Mode: Continuous (will keep running until Ctrl+C)")
    print("=" * 80)

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset=offset_reset,
            enable_auto_commit=True,
            group_id=group_id,
            consumer_timeout_ms=timeout_ms,  # None = run forever
        )
    except Exception as e:
        print(f"\n✗ Failed to connect to Kafka: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Kafka is running: docker ps | grep kafka")
        print("  2. Check connection: telnet localhost 9092")
        print(  # noqa: BLK100
            "  3. Start Kafka: "  # noqa: BLK100
            "docker-compose -f docker-compose-kafka.yml up -d"
        )
        sys.exit(1)

    print("\n✓ Connected successfully!")
    print("Listening for messages... (Press Ctrl+C to stop)\n")
    print("=" * 80)

    message_count = 0
    start_time = datetime.now()

    try:
        for message in consumer:
            message_count += 1
            data = message.value

            timestamp = datetime.now().strftime('%H:%M:%S')  # noqa: BLK100
            print(f"\n📥 Message #{message_count} [{timestamp}]")
            print("─" * 80)
            print("  Kafka Metadata:")
            print(f"    Partition: {message.partition}")
            print(f"    Offset: {message.offset}")
            key_str = (
                message.key.decode('utf-8') if message.key else 'None'
            )
            print(f"    Key: {key_str}")
            msg_time = datetime.fromtimestamp(
                message.timestamp / 1000
            ).strftime('%Y-%m-%d %H:%M:%S')
            print(f"    Timestamp: {msg_time}")
            print("\n  VLM Result:")
            print(f"    Stream ID: {data['stream_id']}")
            start_t = data['segment']['start_time']
            end_t = data['segment']['end_time']
            print(f"    Time Range: {start_t:.1f}s - {end_t:.1f}s")
            print(f"    Duration: {data['segment']['duration']:.1f}s")

            # Format result text with wrapping
            result_text = data["result"]
            if len(result_text) > 200:
                # Wrap long text
                print("    Result:")
                for i in range(0, len(result_text), 100):
                    print(f"      {result_text[i:i+100]}")
            else:
                print(f"    Result: {result_text}")

            print("\n  Metadata:")
            print(f"    Source: {data['metadata']['source']}")
            print(f"    Version: {data['metadata']['version']}")
            print(f"    Publish Time: {data.get('timestamp', 'N/A')}")
            print("─" * 80)

        # If we reach here naturally (not via exception), it's a timeout
        if timeout_ms:
            timeout_secs = timeout_ms / 1000
            print(
                f"\n⏱️  Timeout reached "
                f"({timeout_secs:.0f}s of no new messages)"
            )

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ Error consuming messages: {e}")
        import traceback

        traceback.print_exc()

    consumer.close()

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80)
    print("Session Summary")
    print("=" * 80)
    print(f"  Messages consumed: {message_count}")
    print(f"  Duration: {elapsed:.1f}s")
    if message_count > 0 and elapsed > 0:
        print(f"  Rate: {message_count / elapsed:.2f} messages/sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
