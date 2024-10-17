# DeepStream Parallel Inference

DeepStream Parallel Inference is one application which run inference in parallel. tee will send batched buffer to different models.  nvdsmetamux will mux batch meta from different models which run in parallel. User can configure the pipeline in application configuration yaml file. User can select source ids which need inference in preprocess configuration file. User can select source ids which need mux in nvdsmetamux configuration file.

## Prerequisites
DeepStream SDK 6.1.1 GA or above


## Getting Started:
To get started, please follow these steps.
1. Install [DeepStream](https://developer.nvidia.com/deepstream-sdk) on your platform, verify it is working by running deepstream-app.
2. Install nvdsmetamux.
3. Install triton client video template libraries.
4. Clone the repository preferably in `$DEEPSTREAM_DIR/sources/apps/sample_apps`.
5. Compile and run the program

 ```
  $ cd deepstream-parallel-infer/
  $ sudo make
  $ sudo ./deepstream-parallel-infer -c <config.yml>
```
Run parallel inference app with default configure file: $ ./deepstream-parallel-infer -c source4_1080p_dec_parallel_infer.yml

For any issues or questions, please feel free to make a new post on the [DeepStreamSDK forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/).
