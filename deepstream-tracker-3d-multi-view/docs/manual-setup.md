## Manual Setup Instructions


1. Please check [Deepstream Container Prerequisites](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html#prerequisites) for Deepstream container setup, and download the latest DeepStream container image (e.g., DS 8.0 in the example below). 
    ```bash
    docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch
    ```
2. Git clone the current `deepstream_reference_apps` repository to the host machine and enter `deepstream-tracker-3d-multi-view` directory
    ```bash
    # Install Git LFS
    sudo apt install git-lfs
    git lfs install

    git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git
    cd deepstream_reference_apps/deepstream-tracker-3d-multi-view
    git lfs pull  # In case repo is already cloned before installing git-lfs
    ```

3. Unzip the datasets.zip managed by Git LFS
    ```bash
    unzip assets/datasets.zip
    ```


4. Download the `PeopleNetTransformer` and `BodyPose3DNet` models from NGC, and build the custom parser for `PeopleNetTransformer` model
    * Download the models ([PeopleNetTransformer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer_v2), and [BodyPose3DNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodypose3dnet))
    ```bash
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx' -O 'models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx'
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/bodypose3dnet/deployable_accuracy_onnx_1.0/files?redirect=true&path=bodypose3dnet_accuracy.onnx' -O 'models/BodyPose3DNet/bodypose3dnet_accuracy.onnx'
    ```

    * Build the custom parser
    ```bash
    # Enter the DeepStream container
    docker run -it --privileged --rm --net=host --runtime=nvidia \
        -v $PWD/models:/workspace/models \
        -w /workspace/models/PeopleNetTransformer \
        --entrypoint /bin/bash \
        nvcr.io/nvidia/deepstream:8.0-triton-multiarch

    # Build the custom parser
    cd custom_parser
    make clean && make
    # [Expected output] You should see "libnvds_infercustomparser_tao.so" built under the custom_parser directory. And it is expected to see warnings during the build process.

    # Exit the DeepStream container
    exit
    ```

5. Install and run the Mosquitto MQTT broker

    * Install Mosquitto and its client tools:
    ```bash
    sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
    sudo apt update
    sudo apt install mosquitto mosquitto-clients
    ```

    * After the installation, the Mosquitto broker service will be automatically started on port 1883. You can verify this by running the provided test script. If the broker is active, you should see `Hello from Mosquitto test!` in the output:
    ```bash
    chmod +x ./scripts/mosquitto_test.sh
    ./scripts/mosquitto_test.sh
    ```

    * If the previous step fails (e.g. seeing `Error: Connection refused`), use the following command to start Mosquitto on port 1883, and then run the test script again:
    ```bash
    mosquitto -p 1883
    # [Expected output] You should see "mosquitto version <version_number> running" printed.
    # You need to keep it running in a separate terminal window. To avoid this, you can use the following command to start it in the background:
    # mosquitto -p 1883 -d
    # [Expected output] Nothing will be printed. Use the mosquitto_test.sh script to verify the broker is running.
    # And to kill it, you can use the following command:
    # kill -9 $(lsof -t -i:1883)
    ```

    * Please refer to [Mosquitto documentation](https://mosquitto.org/download/) if you still encounter issues.

6. Install and start a Kafka broker, and create a `mv3dt` topic:
    * Follow the [Kafka quickstart](https://kafka.apache.org/quickstart) to download and start Kafka. The commands are provided below. **Note that please start a separate terminal window to keep the Kafka broker running.**
    ```bash
    # Kafka requires Java 17+. Check your Java version.
    # If you see "Command 'java' not found" or it is older than 17, please install openjdk-17-jdk.
    java -version
    sudo apt install openjdk-17-jdk 

    # Get Kafka
    wget https://dlcdn.apache.org/kafka/4.0.0/kafka_2.13-4.0.0.tgz
    tar -xzf kafka_2.13-4.0.0.tgz
    cd kafka_2.13-4.0.0

    # Start the Kafka environment
    export KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"

    bin/kafka-storage.sh format --standalone -t $KAFKA_CLUSTER_ID -c config/server.properties
    # [Expected output] You should see `Formatting dynamic metadata voter directory /tmp/kraft-combined-logs with metadata.version 4.0-IV3.`

    bin/kafka-server-start.sh config/server.properties
    # [Expected output] You should see `Kafka Server started.` and it will keep logging `INFO` messages.
    ```

    * Create a `mv3dt` topic under broker server `localhost:9092`, and set the message retention to 30 seconds. 

    ```bash
    cd <path to kafka folder, e.g. kafka_2.13-4.0.0>

    ./bin/kafka-topics.sh --bootstrap-server localhost:9092 \
        --create \
        --topic mv3dt \
        --partitions 1 \
        --replication-factor 1 \
        --config retention.ms=30000 \
        --if-not-exists
    # [Expected output] Seeing `Created topic mv3dt.` or nothing if the topic already exists.
    ```
    * After you have followed the above Kafka setup steps, in the future, you only need to run the following command to start Kafka:
    ```bash
    bin/kafka-server-start.sh config/server.properties
    # [Expected output] It is expected to see DUPLICATE_BROKER_REGISTRATION in the logs. As long as the broker keeps running and logging INFO messages, you can proceed.
    ```

    * To stop a Kafka broker running in the background, you can use the following command:
    ```bash
    cd <path to kafka folder, e.g. kafka_2.13-4.0.0>
    bin/kafka-server-stop.sh
    ```


7. Install the required Python dependencies. Note that the scripts in this repo expect a virtual environment named `mv3dt_venv` located under the root of the repo. Please make sure to follow the following instructions exactly for quick start. 

    ```bash
    cd <path to current repo, i.e. deepstream-tracker-3d-multi-view>

    # Install required deb packages
    sudo apt update
    sudo apt install python3-tk python3.12-venv python3.12-dev

    # Create a python virtual enviornment named `mv3dt_venv` and install required python packages
    python3 -m venv mv3dt_venv
    source mv3dt_venv/bin/activate

    pip install -r requirements.txt
    ```
    * Check the virtual environment. If any specific package fails, please install it manually with `pip install <package-name>`.
    ```bash
    ls -d mv3dt_venv
    # [Expected output] You should see "mv3dt_venv" printed. If you see "No such file or directory", please check the previous step "python3 -m venv mv3dt_venv".

    pip list
    # [Expected output] You should see kafka-python, protobuf in the list
    ```


8. (Optional) This step is only needed if you choose to use Option 2: Inference Builder.

    Set up [Deepstream Inference Builder](https://github.com/NVIDIA-AI-IOT/inference_builder). It is recommended to clone the `inference_builder` repo outside of the current repo.
    * Clone the inference builder repo

    ```bash
    git clone https://github.com/NVIDIA-AI-IOT/inference_builder.git
    cd inference_builder
    git submodule update --init --recursive
    ```
    * Create a new virtual environment for inference builder and install prerequisites. Please follow the following instructions exactly for quick start. **Note that there are 2 virtual environments used in this repo, `mv3dt_venv` and `ib_venv`. The scripts provided in the repo assumes that a `mv3dt_venv` folder is under the current repo, and a `ib_venv` folder is under the inference_builder repo.**

    ```bash
    # Install required deb packages
    sudo apt install protobuf-compiler

    # Deactivate the mv3dt_venv, and create a new virtual environment named ib_venv for inference builder
    deactivate
    python -m venv ib_venv
    source ib_venv/bin/activate
    pip3 install -r requirements.txt
    ```
    * Check the virtual environment. If any specific package fails, please install it manually with `pip install <package-name>`.
    ```bash
    ls -d ib_venv
    # [Expected output] You should see "ib_venv" printed. If you see "No such file or directory", please check the previous step "python -m venv ib_venv". 

    pip list
    # [Expected output] You should see omegaconf  2.3.0 in the list
    ```
    * Build a Docker image named `inference-builder-mv3dt:latest` with Inference Builder python dependencies.
    ```bash
    # Create a temporary Dockerfile
    cat > ./Dockerfile.ib_mv3dt << 'EOF'
    FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch
    RUN pip3 install torch==2.7.0 omegaconf==2.3.0
    ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
    ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
    ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
    WORKDIR /mv3dt_app
    EOF

    # [Expected output] You should see a Dockerfile.ib_mv3dt file created under the current directory.

    # Build the Docker image
    docker build -f ./Dockerfile.ib_mv3dt -t inference-builder-mv3dt:latest .

    # [Expected output] You should see "naming to docker.io/library/inference-builder-mv3dt:latest" printed as the last line.
    ```

