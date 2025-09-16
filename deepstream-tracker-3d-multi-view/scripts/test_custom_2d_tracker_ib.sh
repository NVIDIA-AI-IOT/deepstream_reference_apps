# Set dataset, model, experiment, and repo directories
export DATASET_DIR=$PWD/datasets/mtmc_6cam/
export EXPERIMENT_DIR=$PWD/experiments/inference_builder/6cam
export MODEL_REPO=$PWD/models

export REPO_DIR=$PWD
export INFERENCE_BUILDER_DIR=${INFERENCE_BUILDER_DIR:-$HOME/inference_builder}

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

# Generate Inference Builder configuration files with custom 2D tracker config
source mv3dt_venv/bin/activate
python utils/inference_builder_auto_configurator.py \
    --dataset-dir=$DATASET_DIR \
    --tracker-config=config_tracker_2d.yml \
    --config-overrides=override_tracker_12cam.yml \
    --output-dir=$EXPERIMENT_DIR

cp $EXPERIMENT_DIR/nvdsinfer_config.yaml $MODEL_REPO/PeopleNetTransformer/

# Generate Python package for MV3DT inference flow using Inference Builder
cd $INFERENCE_BUILDER_DIR
source ib_venv/bin/activate
python builder/main.py $EXPERIMENT_DIR/ds_mv3dt.yaml \
    -o builder/samples/mv3dt_app \
    --server-type serverless

# Launch real-time BEV visualization
cd $REPO_DIR
source mv3dt_venv/bin/activate
python utils/kafka_bev_visualizer.py \
    --dataset-path=$DATASET_DIR \
    --msgconv-config=$EXPERIMENT_DIR/config_msgconv.txt \
    --average-multi-cam \
    --show-ids &

# Launch MV3DT pipeline with Inference Builder and custom 2D tracker config
docker run --privileged --rm -it --net=host $GPU_FLAG \
    -v $INFERENCE_BUILDER_DIR/builder/samples/mv3dt_app/deepstream-app:/mv3dt_app \
    -v $MODEL_REPO:/workspace/models \
    -v $DATASET_DIR:/workspace/inputs \
    -v $EXPERIMENT_DIR:/workspace/experiments \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /mv3dt_app \
    inference-builder-mv3dt:latest \
    python3 __main__.py --source-config /workspace/experiments/source_list_static.yaml -s /dev/null
