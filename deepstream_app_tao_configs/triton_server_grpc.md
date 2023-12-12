# Triton Server
## [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) Bring Up

DeepStream applications can work as Triton Inference client. So the corresponding Triton Inference Server should be started before the Triton client start to work.

An immediate way to start a corresponding Triton Server is to use Triton containers provided in [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). Since every DeepStream version has its corresponding Triton Server version, so the reliable way is to use the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream).

* The Triton Server can be started in the same machine which the DeepStream application works in, please make sure the Triton Server is started in a new terminal.

* The Triton Server can be started in another machine as the server which is coonected to the machine for DeepStream applications through ehternet. 

## Prepare Triton Server For gRPC Connection
The following steps take the DeepStream 6.4 GA as an example, if you use other DeepStream versions, the corresponding DeepStream Triton [image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) can be used.

To start Triton Server with DeepStream Triton container, the docker should be run in a new terminal and the following commands should be run in the same path as the deepstream_app_tao_configs codes are downloaded:
* Start the Triton Inferece Server with DeepStream Triton docker
```
    //start Triton docker, 10001:8001 is used to map docker container's 8000 port to host's 10000 port, these ports can be changed.
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -p 10000:8000 -p 10001:8001 -p 10002:8002  -v $(pwd):/samples   -e DISPLAY=$DISPLAY -w /samples nvcr.io/nvidia/deepstream:6.4-gc-triton-devel
```

Then the model engines should be generated in the server, the [tao-converter links](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter) inside the prepare_triton_models.sh script can be changed to proper versions according to the actual TensorRT version:

```
    ./prepare_triton_models.sh
```

If the server is running in the same machine as the DeepStream application, the following command can be used directly. If it is not, please set the gRPC url as the IP address of the server machine in all the configuration files in deepstream_app_tao_configs/triton-grpc:

The gRPC url setting looks like:
```
grpc {
        url: "192.168.0.51:10001"
    }
```

Then the Triton Server service can be started with the following command:
```
    tritonserver --model-repository=/samples/triton --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose=1 --exit-on-error=false

```

The DeepStream sample application should run in another terminal with the Triton Inference client libraries installed. It is recommend to run the application in the DeepStream Triton container, please refer to [triton_server.md](./triton_server.md) for how to start a DeepStream Triton container.
