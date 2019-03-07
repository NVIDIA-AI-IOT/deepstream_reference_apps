
# Squeeze-and-Excitation Networks Reference Apps using TensorRT 5 and DeepStream SDK 3.0

## Squeeze-and-Excitation Networks ##

1. Introductions and implementations from the author: [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) and [Repository](https://github.com/hujie-frank/SENet)

2. The TensorFlow re-implementation from which the checkpoints and TensorBoard are extracted for TensorRT inference: [Repository](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet)

## Getting Started ##
This repository provides the SENet reference applications using TensorRT 5 and DeepStream SDK 3.0.

There are two sample applications under `senet/apps` : trt-sent and deeptream-resnet-senet.

1. `trt-senet-app` is the stand alone application using TensorRT and SENet.

    Further details are provided in [here](apps/trt-senet/README.md)<br/>
    To use only the stand alone trt-senet-app, Deepstream installation can be skipped.<br/>

2. `deepstream-senet-app` is the reference application using Deepstream and SENet.

    SENet is used as a secondary inference in this example.<br/>
    In order to run this app, you must run `trt-senet-app` to obtain TensorRT engine for SENet.<br/>
    In addition, you must install Deepstream and other pre-requisites.<br/>
    Further details are provided in [here](apps/deepstream-senet/README.md)<br/>
