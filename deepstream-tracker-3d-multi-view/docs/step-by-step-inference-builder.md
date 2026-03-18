# Inference Builder: Step-by-step Instructions

This page provides detailed step-by-step instructions for running MV3DT using Inference Builder. For quick start scripts, see the [main README](../README.md#option-2-running-mv3dt-using-inference-builder).

## Sample 1: 4-camera dataset

1. Set up environment variables and prepare experiment directories
    ```bash
    export DATASET_DIR=$PWD/datasets/mtmc_4cam/
    export EXPERIMENT_DIR=$PWD/experiments/inference_builder/4cam
    export MODEL_REPO=$PWD/models

    mkdir -p $EXPERIMENT_DIR/infer-kitti-dump
    mkdir -p $EXPERIMENT_DIR/tracker-kitti-dump
    ```

2. Generate Inference Builder configuration files using the auto-configurator
    
    ```bash
    # Activate the Python environment
    source mv3dt_venv/bin/activate
    
    # Generate configs with 4-camera overrides
    python utils/inference_builder_auto_configurator.py \
        --dataset-dir=$DATASET_DIR \
        --config-overrides=override_tracker_4cam.yml \
        --output-dir=$EXPERIMENT_DIR
    # [Expected output] You should see 
    #   Generated files:
    #     - ds_mv3dt.yaml (inference config with max_batch_size: 4)
    #     - config_tracker.yml (3D tracker config)
    #     - source_list_static.yaml (source configuration)
    #     - nvdsinfer_config.yaml (inference engine config with batch_size: 4)
    #     - config_msgconv.txt (message converter config)
    #     - pub_sub_info_config_0.yml (communication config)

    # Copy the generated nvdsinfer config to the model directory
    cp $EXPERIMENT_DIR/nvdsinfer_config.yaml $MODEL_REPO/PeopleNetTransformer/
    ```

3. Generate a Python package at `$INFERENCE_BUILDER_DIR/builder/samples/mv3dt_app` containing the MV3DT inference flow.
    ```bash
    export INFERENCE_BUILDER_DIR=<path to inference builder repo>
    cd $INFERENCE_BUILDER_DIR
    source ib_venv/bin/activate
    python builder/main.py $EXPERIMENT_DIR/ds_mv3dt.yaml \
        -o builder/samples/mv3dt_app \
        --server-type serverless
    ```

4. (Optional) Launch real-time BEV visualization.
    Please keep it running in a separate terminal window or add `&` to the end of the command to run it in the background.
    
    ```bash
    # Return to the repo directory and activate the Python environment
    cd <path to current repo, i.e. deepstream-tracker-3d-multi-view>
    source mv3dt_venv/bin/activate
    
    # Start BEV visualization
    python utils/kafka_bev_visualizer.py \
        --dataset-path=$DATASET_DIR \
        --msgconv-config=$EXPERIMENT_DIR/config_msgconv.txt \
        --average-multi-cam \
        --show-ids

    # [Expected output] You should see a window named "Bird-Eye View of Multi-View 3D Tracking" pop up and will display the live tracking results. 
    # Select the window and press 'q' to quit.
    ```

5. Launch the `inference-builder-mv3dt:latest` container with volume mounts, including the Python package generated in the previous step.

    Note that this container is built during prerequisites setup. Please refer to the Inference Builder setup step in [Manual Setup Instructions](manual-setup.md) for more details.
    ```bash
    sudo xhost + # give container access to display

    docker run --privileged --rm -it --net=host --runtime=nvidia \
    -v $INFERENCE_BUILDER_DIR/builder/samples/mv3dt_app/deepstream-app:/mv3dt_app \
    -v $MODEL_REPO:/workspace/models \
    -v $DATASET_DIR:/workspace/inputs \
    -v $EXPERIMENT_DIR:/workspace/experiments \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /mv3dt_app \
    inference-builder-mv3dt:latest \
    python3 __main__.py --source-config /workspace/experiments/source_list_static.yaml -s /dev/null

    # [Expected output] You should see a window named "python3" pop up and will display 4 camera views in a grid.
    # Run this command to quit early: `docker ps -q --filter "ancestor=inference-builder-mv3dt" | xargs docker stop`.
    # By default, the application waits up to **1000 seconds** if there is no data being streamed before exiting gracefully. You should see "Inference completed" as the last line from the logs.
    ```


## Sample 2: 12-camera dataset

The steps are the same as for the 4-camera dataset, except setting `DATASET_DIR` and `EXPERIMENT_DIR` to the 12-camera directories.

```bash
export DATASET_DIR=$PWD/datasets/mtmc_12cam/
export EXPERIMENT_DIR=$PWD/experiments/inference_builder/12cam

python utils/inference_builder_auto_configurator.py \
    --dataset-dir=$DATASET_DIR \
    --output-dir=$EXPERIMENT_DIR
```
