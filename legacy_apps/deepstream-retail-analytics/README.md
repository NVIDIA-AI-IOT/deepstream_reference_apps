# Description

This is a sample application to perform real-time Intelligent Video Analytics (IVA) in a brick and mortar retail environment using NVIDIA DeepStream, TAO, and pre-trained models. DeepStream is used to run DL inference on a video feed inside a store to detect and track customers, and identify whether the detected persons are carrying shopping baskets. The inference output of this Computer Vision (CV) pipeline is streamed, using Kafka, to a Time-Series Database (TSDB) for archival and further processing. A Django app serves a REST-ful API to query insights based on the inference data. We also demonstrate a sample front-end dashboard to quickly visualize the various Key Performance Indicators (KPIs) available through the Django app.

This application is based on deepstream-test4 and deepstream-test5 sample applications included with DeepStream. The architecture diagram below shows how all the components are connected.

![](./media/output.gif)

What is this DeepStream pipeline made of?

* Primary Detector: PeopleNet Pre-Trained Model (PTM) from NGC
* Secondary Detector: Custom classification model trained using TAO toolkit to classify people with and without shopping baskets
* Object Tracker: NvDCF tracker
* Message Converter: Custom message converter to generate custom payload from inference data
* Message Broker: Message broker to relay inference data to a kafka server

# Table of Contents
* [Description](#description)
* [Table of Contents](#table-of-contents)
* [Application Architecture](#application-architecture)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
* [Build](#build)
* [Running the Application](#run-the-application)
* [Output](#output)
* [Advanced](#advanced)

# Application Architecture


   <center> <img src="./media/retail-iva-arch.png" width=500></center>


# Quick Start

# Prerequisites

1. Install the latest [NVIDIA drivers](https://www.nvidia.com/download/index.aspx) for your operating system and GPU.

2. Install Docker and the NVIDIA Container Toolkit - Refer to this [README](docs/install_nvidia_container_toolkit.md).

3. **OPTIONAL:** Install python and pip. Required for front-end only. Can omit if not using front-end.

4. Install DeepStream SDK [instructions](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

    * Pull the docker image for DeepStream development
    ```bash
    docker pull nvcr.io/nvidia/deepstream:6.1-devel
    ```

    * Allow external applications to connect to the host's X display
    ```bash
    xhost +
    ```

    **Note:** If you are using a remote machine, the above command will not work from an SSH session. It has to be executed from a VNC/RDP connection.

    * Run the container
    ```bash
    docker run -it --entrypoint /bin/bash --gpus all --rm --network=host -e DISPLAY=:0 -v /tmp/.X11-unix/:/tmp/.X11-unix --privileged -v /var/run/docker.sock:/var/run/docker.sock nvcr.io/nvidia/deepstream:6.1-devel
    ```

    This command will
    * Start the container
    * Provide access to all GPUs
    * Host the container on the host's network
    * Forwards the display of the host to the container along with some other volumes
    * Opens an interactive terminal to run commands from within the container

5. Install git-lfs inside the container

    ```bash
    apt install git-lfs
    ```


6. We need a kafka message broker and kSQL database. For the purpose of this project, we use [confluent-platform](https://docs.confluent.io/platform/current/quickstart/ce-docker-quickstart.html) to setup these services. So, lets setup confluent-platform:

    **Note:** Bash commands in this section should be run from a separete terminal window and **not from within the DeepStream container**.

    ```bash
    wget https://raw.githubusercontent.com/confluentinc/cp-all-in-one/7.2.1-post/cp-all-in-one/docker-compose.yml
    docker-compose up -d
    ```
    * Verify if all the containers started successfully by running `docker ps`

    ![](./media/docker_container_ps.png)

7. Create a kafka-topic that will be used to receive messages sent by the DeepStream app

    ```bash
    docker exec -it broker /bin/bash
    # Within the container
    kafka-topics --bootstrap-server "localhost:9092" --topic "detections" --create
    ```

    * You can also create the kafka topic by navigating to the confluent control center > cluster > Topics > Add Topic.

8. Setup a [kSQL stream](https://docs.ksqldb.io/en/latest/concepts/streams/) based on the topic `detections`:

    a) If you used the above mentioned docker compose file, you can access kSQL CLI by running
    ```
    docker exec -it ksqldb-cli ksql http://ksqldb-server:8088
    ```
    b) Once the CLI is active, copy-paste the content from [confluent-platform/stream_creation.sql](confluent-platform/stream_creation.sql) into the CLI to create the stream
    
    **Note:** You don't have to explicitly create the topic in confluent-kafka. The broker will automatically create a topic once DeepStream sends messages to a new topic.

# Getting Started

1. If you are using DeepStream via a docker container as mentioned in the instructions above, execute the following command to open the terminal of the DeepStream docker container if it's not already open. Otherwise, skip this step.

    ```bash
    docker exec -it <container_id> /bin/bash
    ```

    You can locate the container id by running the following command:

    ```bash
    docker container ps
    ```
    ![](./media/deepstream_container.png)

2. Clone the repo in $DS_SDK_ROOT/sources/apps/sample_apps/
    ```bash
    cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps
    git clone https://github.com/NVIDIA-AI-IOT/deepstream-retail-analytics.git
    cd deepstream-retail-analytics
    git lfs pull
    ```

    * Although not necessary, it is recommended to verify the checksum of model and input files to confirm file integrity
    ```bash
    cd files/
    sha512sum -c checksum.txt
    ```

3. Download PeopleNet model with the model download script. Download the `.etlt` and `labels.txt` files.
    ```bash
    bash ./download_models.sh
    ```
4. The custom message converter should be built inside the docker. The build instructions are [custom nvmsgconv library](nvmsgconv/README.md).

# Build for x86 dGPU system

Run the following commands from project root

Modify the below command with the cuda version installed in the docker container. To check the CUDA version inside the docker container you can use `nvcc --version` command.

```bash
export CUDA_VER=<cuda_version>
make -B
```

# Run the application

## Running the DeepStream Application

```bash
./ds-retail-iva configs/retail_iva.yml --no-display
```

The `--no-display` flag in the above command is optional. If the application is running from within a docker container without a display attached, you should use this flag.

## Running the front-end

```bash
cd ds-retail-iva-frontend
pip install -r requirements.txt
python3 manage.py runserver 0.0.0.0:8000
```

Open a browser and go to [http://localhost:8000](http://localhost:8000) to visualize the dashboard

# Output

**Dashboard**

<img src="./media/dashboard.png" width="1000">

# Advanced

[TAO README](./TAO/README.md) - Follow the instructions in this file to create a dataset and train a classification model using TAO toolkit

[NvMsgConv README](./nvmsgconv/README.md) - Follow this README to build a custom library to modify message payload generated by DeepStream
