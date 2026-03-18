# DeepStream Container: Step-by-step Instructions

This page provides detailed step-by-step instructions for running MV3DT using the DeepStream Container. For quick start scripts, see the [main README](../README.md#option-1-running-mv3dt-using-deepstream-container).

## Sample 1: 4-camera dataset

1. Set up environment variables and prepare experiment directories
    ```bash
    export DATASET_DIR=$PWD/datasets/mtmc_4cam/
    export EXPERIMENT_DIR=$PWD/experiments/deepstream/4cam
    export MODEL_REPO=$PWD/models

    mkdir -p $EXPERIMENT_DIR/infer-kitti-dump
    mkdir -p $EXPERIMENT_DIR/tracker-kitti-dump
    mkdir -p $EXPERIMENT_DIR/outVideos
    ```
2. Generate DeepStream configuration files using the auto-configurator
   
   The auto-configurator automatically generates all necessary configuration files based on your dataset. It supports various output options (OSD display, video file output, Kafka streaming) and can work with both 2D and 3D tracker configurations. 
   
   **About Override Files:** The `--config-overrides` parameter allows you to apply dataset-specific settings. For example, `override_tracker_4cam.yml` is optimized for the sample 4-camera dataset (which uses feet as world coordinate units). You can create custom override files for your own datasets.
   
   For more info on the auto-configurator, see [`utils/README.md`](../utils/README.md#deepstream_auto_configuratorpy).
   
    ```bash
    # Activate the Python environment
    source mv3dt_venv/bin/activate
    
    # Generate configs with 4-camera overrides
    python utils/deepstream_auto_configurator.py \
        --dataset-dir=$DATASET_DIR \
        --enable-msg-broker \
        --enable-osd \
        --config-overrides=override_tracker_4cam.yml \
        --output-dir=$EXPERIMENT_DIR
    
    # [Expected output] You should see
    #   Generated files:
    #     - config_deepstream.txt (main pipeline config)
    #     - config_tracker.yml (3D tracker config)
    #     - config_msgconv.txt (message converter config)
    #     - pub_sub_info_config_0.yml (communication config)

    ```
3. (Optional) Launch real-time BEV visualization
    
    Before launching the main MV3DT pipeline, optionally start the bird's-eye view visualizer to see real-time 3D tracking results. Please keep it running in a separate terminal window or add `&` to the end of the command to run it in the background.
    
    ```bash
    # Start BEV visualization
    python utils/kafka_bev_visualizer.py \
        --dataset-path=$DATASET_DIR \
        --msgconv-config=$EXPERIMENT_DIR/config_msgconv.txt \
        --average-multi-cam \
        --show-ids 

    # [Expected output] You should see a window named "Bird-Eye View of Multi-View 3D Tracking" pop up and will display the live tracking results.
    # Select the window and press 'q' to quit.
    ```

4. Launch MV3DT

    The following command mounts the necessary folders into the DeepStream container and starts the `deepstream-test5-app` with MV3DT configs. 

    ```bash
    sudo xhost + # give container access to display

    docker run -t --privileged --rm --net=host --runtime=nvidia \
        -v $MODEL_REPO:/workspace/models \
        -v $DATASET_DIR:/workspace/inputs \
        -v $EXPERIMENT_DIR:/workspace/experiments \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        -w /workspace/experiments \
        nvcr.io/nvidia/deepstream:9.0-triton-multiarch \
        deepstream-test5-app -c config_deepstream.txt

    # [Expected output] You should see a window named "DeepStreamTest5App" pop up and will display 4 camera views in a grid.
    # Select the window and press 'q' to quit early.
    # The pipeline will quit automatically with "App run succesful" as the last line from the logs.
    ```

## Sample 2: 12-camera dataset

The steps are the same as for the 4-camera dataset, except setting `DATASET_DIR` and `EXPERIMENT_DIR` to the 12-camera directories. The auto-configurator automatically detects the number of cameras in your dataset and generates required config files for 12-camera dataset.

```bash
export DATASET_DIR=$PWD/datasets/mtmc_12cam/
export EXPERIMENT_DIR=$PWD/experiments/deepstream/12cam

python utils/deepstream_auto_configurator.py \
    --dataset-dir=$DATASET_DIR \
    --enable-msg-broker \
    --enable-osd \
    --output-dir=$EXPERIMENT_DIR
```
