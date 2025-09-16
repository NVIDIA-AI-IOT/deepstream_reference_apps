#!/bin/bash

# Test parameters
BROKER="localhost"      # Change if the broker is on a different machine or IP address
PORT=1883              # Default MQTT port for Mosquitto
TOPIC="test/topic"     # The topic to test
MESSAGE="Hello from Mosquitto test!"  # The test message


# Subscribe to the topic in the background
echo "Subscribing to topic '$TOPIC'..."
mosquitto_sub -h $BROKER -p $PORT -t $TOPIC & 
SUB_PID=$!

# Give the subscriber some time to start up
sleep 1

# Publish a message to the topic
echo "Publishing message to topic '$TOPIC'..."
mosquitto_pub -h $BROKER -p $PORT -t $TOPIC -m "$MESSAGE"

# Wait a few seconds to allow the subscriber to receive the message
sleep 2

# Stop the mosquitto_sub process
kill $SUB_PID
wait $SUB_PID

# The script will automatically exit, and the message should be received by the subscriber.
echo "Test completed."
