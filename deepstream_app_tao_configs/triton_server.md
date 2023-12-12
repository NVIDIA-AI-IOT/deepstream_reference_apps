## Prepare Triton Server For Native Inferencing
As mentioned in the README, the DeepStream applications should work as Triton client with Triton Server running natively for cAPIs. So the [Triton Inference Server libraries](https://github.com/triton-inference-server/client) should be installed in the machine. An easier way is to run the DeepStream application in the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream). 

Running DeepStream Triton container, takes the DeepStream 6.1 GA container as the example:
```
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/samples   -e DISPLAY=$DISPLAY -w /samples nvcr.io/nvidia/deepstream:6.4-gc-triton-devel
```
Inside the container, prepare model engines for Triton server, the [tao-converter links](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter) inside the prepare_triton_models.sh scripts can be changed to proper versions according to the actual TensorRT version:
```
    ./prepare_triton_models.sh

```

Then the DeepStream sample application can be build and run inside this container.
