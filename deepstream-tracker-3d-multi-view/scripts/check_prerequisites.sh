#!/bin/bash

# Global variables - same as setup script
BASE_DIR=${BASE_DIR:-$HOME}
USE_INFERENCE_BUILDER=${USE_INFERENCE_BUILDER:-false}
KAFKA_VERSION="4.0.0"
SCALA_VERSION="2.13"

# Standardized paths
KAFKA_DIR="$BASE_DIR/kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
INFERENCE_BUILDER_DIR="$BASE_DIR/inference_builder"

# Initialize status variables
GPU_STATUS="✓"
MQTT_STATUS="✓"
KAFKA_STATUS="✓"
MV3DT_VENV_STATUS="✓"
INFERENCE_BUILDER_STATUS="✓"
DATASETS_MODELS_STATUS="✓"

# Check 1: Check if NVIDIA GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU is not available"
    GPU_STATUS="✗"
fi

# Check 2: Check if MQTT broker is running on port 1883
if ! ./scripts/mosquitto_test.sh > /dev/null 2>&1; then
    MQTT_STATUS="✗"
fi

# Check 3: Check if Kafka is running
if [ ! -d "$KAFKA_DIR" ]; then
    echo "Kafka directory not found: $KAFKA_DIR"
    KAFKA_STATUS="✗"
else
    KAFKA_TOPICS_SCRIPT="$KAFKA_DIR/bin/kafka-topics.sh"
    if [ -x "$KAFKA_TOPICS_SCRIPT" ]; then
        TOPIC_LIST=$(timeout 5s $KAFKA_TOPICS_SCRIPT --bootstrap-server localhost:9092 --list 2>/dev/null)
        if ! echo "$TOPIC_LIST" | grep -q "mv3dt"; then
            echo "Kafka topics: $TOPIC_LIST"
            echo "mv3dt topic not found"
            KAFKA_STATUS="✗"
        fi
    else
        echo "Kafka topics script not found or not executable: $KAFKA_TOPICS_SCRIPT"
        KAFKA_STATUS="✗"
    fi
fi

# Check 4: Check whether virtual env "mv3dt_venv" setup correctly
if [ ! -d "$PWD/mv3dt_venv" ]; then
    echo "Virtual environment 'mv3dt_venv' not found"
    MV3DT_VENV_STATUS="✗"
fi

# Check 5: Check whether inference builder is setup correctly
if [[ "$USE_INFERENCE_BUILDER" != "true" ]]; then
    INFERENCE_BUILDER_STATUS="(skipped)"
else
    if [ ! -d "$INFERENCE_BUILDER_DIR" ]; then
        echo "Inference builder directory not found: $INFERENCE_BUILDER_DIR"
        INFERENCE_BUILDER_STATUS="✗"
    elif [ ! -d "$INFERENCE_BUILDER_DIR/ib_venv" ]; then
        echo "Inference builder virtual environment 'ib_venv' not found: $INFERENCE_BUILDER_DIR/ib_venv"
        INFERENCE_BUILDER_STATUS="✗"
    fi
fi

# Check 6: check wether datasets and models are setup correctly
if [ ! -d "$PWD/datasets" ]; then
    echo "Datasets directory not found"
    DATASETS_MODELS_STATUS="✗"
else
    if [ ! -d "$PWD/datasets/mtmc_4cam/camInfo" ]; then
        echo "datasets/mtmc_4cam/camInfo directory not found"
        DATASETS_MODELS_STATUS="✗"
    fi
    if [ ! -d "$PWD/datasets/mtmc_4cam/videos" ]; then
        echo "datasets/mtmc_4cam/videos directory not found"
        DATASETS_MODELS_STATUS="✗"
    fi
    if [ ! -d "$PWD/datasets/mtmc_12cam/camInfo" ]; then
        echo "datasets/mtmc_12cam/camInfo directory not found"
        DATASETS_MODELS_STATUS="✗"
    fi
    if [ ! -d "$PWD/datasets/mtmc_12cam/videos" ]; then
        echo "datasets/mtmc_12cam/videos directory not found"
        DATASETS_MODELS_STATUS="✗"
    fi
fi

if [ ! -d "$PWD/models" ]; then
    echo "Models directory not found"
    DATASETS_MODELS_STATUS="✗"
else
    if [ ! -d "$PWD/models/BodyPose3DNet" ]; then
        echo "BodyPose3DNet model not found"
        DATASETS_MODELS_STATUS="✗"
    fi
    if [ ! -d "$PWD/models/PeopleNetTransformer" ]; then
        echo "PeopleNetTransformer model not found"
        DATASETS_MODELS_STATUS="✗"
    fi
    if [ ! -f "$PWD/models/PeopleNetTransformer/custom_parser/libnvds_infercustomparser_tao.so" ]; then
        echo "PeopleNetTransformer custom parser libnvds_infercustomparser_tao.so not found"
        DATASETS_MODELS_STATUS="✗"
    fi
fi

# Summary of all checks
echo ""
echo "---- PREREQUISITES CHECK SUMMARY ----"
echo "1. NVIDIA GPU:           $GPU_STATUS"
echo "2. MQTT Broker:          $MQTT_STATUS"
echo "3. Kafka Broker:         $KAFKA_STATUS"
echo "4. Python Environment:   $MV3DT_VENV_STATUS"
echo "5. Datasets & Models:    $DATASETS_MODELS_STATUS"
if [[ "$USE_INFERENCE_BUILDER" == "true" ]]; then
    echo "6. Inference Builder:    $INFERENCE_BUILDER_STATUS"
fi
echo "-------------------------------------"

# Check overall status and exit with appropriate code
inference_check_passed=true
if [[ "$USE_INFERENCE_BUILDER" == "true" && "$INFERENCE_BUILDER_STATUS" != "✓" ]]; then
    inference_check_passed=false
fi

if [[ "$GPU_STATUS" = "✓" && "$MQTT_STATUS" = "✓" && "$KAFKA_STATUS" = "✓" && 
      "$MV3DT_VENV_STATUS" = "✓" && "$DATASETS_MODELS_STATUS" = "✓" && 
      "$inference_check_passed" = true ]]; then
    echo ""
    echo "✅ All prerequisites are properly set up!"
    exit 0
else
    echo ""
    echo "❌ Some prerequisites are missing or not properly configured."
    echo "Please run the setup script or follow manual setup instructions."
    exit 1
fi