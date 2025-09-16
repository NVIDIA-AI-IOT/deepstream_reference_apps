# Set dataset, model, and experiment directories
export DATASET_DIR=$PWD/datasets/mtmc_12cam/
export EXPERIMENT_DIR=$PWD/experiments/deepstream/12cam
export MODEL_REPO=$PWD/models

# Set correct GPU flag considering diffent platforms
if docker info | grep -q 'Runtimes.*nvidia'; then
    GPU_FLAG="--runtime=nvidia"
elif docker run --help | grep -q -- "--gpus"; then
    GPU_FLAG="--gpus all"
else
    echo "No GPU support found in Docker."
    exit 1
fi

# Create directories for output
mkdir -p $EXPERIMENT_DIR/infer-kitti-dump
mkdir -p $EXPERIMENT_DIR/tracker-kitti-dump
mkdir -p $EXPERIMENT_DIR/outVideos

# Auto-generate DeepStream configuration files
source mv3dt_venv/bin/activate
python utils/deepstream_auto_configurator.py \
    --dataset-dir=$DATASET_DIR \
    --enable-osd \
    --enable-msg-broker \
    --config-overrides=override_tracker_12cam.yml \
    --output-dir=$EXPERIMENT_DIR

# Launch real-time BEV visualization
python utils/kafka_bev_visualizer.py \
    --dataset-path=$DATASET_DIR \
    --msgconv-config=$EXPERIMENT_DIR/config_msgconv.txt \
    --average-multi-cam \
    --show-ids &

# Launch MV3DT pipeline
docker run -t --privileged --rm --net=host $GPU_FLAG \
    -v $MODEL_REPO:/workspace/models \
    -v $DATASET_DIR:/workspace/inputs \
    -v $EXPERIMENT_DIR:/workspace/experiments \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /workspace/experiments \
    ${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:8.0-triton-multiarch} \
    deepstream-test5-app -c config_deepstream.txt