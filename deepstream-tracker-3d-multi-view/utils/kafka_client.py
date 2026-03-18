"""Simple Kafka client for receiving and decoding MV3DT protobuf messages."""

import json
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from google.protobuf.json_format import MessageToDict
from schema_pb2 import Frame
import argparse

def main():
    """Main function to consume and print Kafka messages."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='mv3dt', help='Kafka topic to subscribe to')
    parser.add_argument('--broker', type=str, default='localhost:9092', help='Kafka broker server (e.g., localhost:9092)')
    args = parser.parse_args()

    topic = args.topic
    broker = args.broker

    print(f"Starting Kafka client reading from broker '{broker}' topic '{topic}'...")

    try:
        # Create Kafka consumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=broker,
            auto_offset_reset='earliest',
            value_deserializer=lambda x: x  # Keep as bytes for protobuf
        )
        
        print("Connected to Kafka. Waiting for messages...")
        
        message_count = 0
        
        # Consume messages
        for msg in consumer:
            try:
                # Parse protobuf message
                frame = Frame()
                frame.ParseFromString(msg.value)
                # print ('frame', frame)
                # Convert to dictionary and then to JSON
                frame_dict = MessageToDict(frame)
                json_str = json.dumps(frame_dict, indent=2)

                message_count += 1
                print(f"\n--- Message {message_count} ---")
                # print(f"Frame ID: {frame_id}")
                # print(f"Sensor ID: {sensor_id}")
                # print(f"Objects: {object_count}")
                print(f"JSON Data:")
                print(json_str)
                print("-" * 50)

            except Exception as e:
                print(f"Error parsing message: {e}")
                continue

    except KafkaError as e:
        print(f"Kafka error: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Consumer stopped")


if __name__ == "__main__":
    main()
