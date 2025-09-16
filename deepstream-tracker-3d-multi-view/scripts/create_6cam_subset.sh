#!/bin/bash

# Simple script to create a random 6-camera subset from 12-camera dataset

set -e

DATASET_DIR="datasets"
SOURCE_DIR="$DATASET_DIR/mtmc_12cam"
TARGET_DIR="$DATASET_DIR/mtmc_6cam"

# Check if source dataset exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source dataset not found: $SOURCE_DIR"
    exit 1
fi

# Create target directory
if [[ -d "$TARGET_DIR" ]]; then
    echo "Target directory already exists: $TARGET_DIR"
    read -p "Remove existing directory and continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
    else
        exit 1
    fi
fi

echo "Creating 6-camera subset from 12-camera dataset..."

# Create directory structure
mkdir -p "$TARGET_DIR/camInfo"
mkdir -p "$TARGET_DIR/videos"

# Randomly select 6 cameras from 1-12
selected_cams=($(seq -f "%03g" 1 12 | shuf -n 6 | sort))

echo "Selected cameras: ${selected_cams[@]}"

# Copy selected camera files
for cam in "${selected_cams[@]}"; do
    cam_file="Warehouse_Synthetic_Cam${cam}"
    
    # Copy camera info file
    if [[ -f "$SOURCE_DIR/camInfo/${cam_file}.yml" ]]; then
        cp "$SOURCE_DIR/camInfo/${cam_file}.yml" "$TARGET_DIR/camInfo/"
        echo "Copied: camInfo/${cam_file}.yml"
    fi
    
    # Copy video file
    if [[ -f "$SOURCE_DIR/videos/${cam_file}.mp4" ]]; then
        cp "$SOURCE_DIR/videos/${cam_file}.mp4" "$TARGET_DIR/videos/"
        echo "Copied: videos/${cam_file}.mp4"
    fi
done

# Copy common files
cp "$SOURCE_DIR/map.png" "$TARGET_DIR/"
cp "$SOURCE_DIR/transforms.yml" "$TARGET_DIR/"

echo ""
echo "✅ Successfully created 6-camera dataset at: $TARGET_DIR"
echo "📁 Selected cameras: ${selected_cams[@]}"
echo ""
echo "To use this subset, update your experiment config to point to: $TARGET_DIR"
