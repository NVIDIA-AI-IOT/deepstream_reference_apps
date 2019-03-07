
# Squeeze-and-Excitation Networks TensorRT Inference on Nvidia GPU #

## Installing Pre-requisites: ##

To use the stand alone trt-se-resnet50-app,

- Prepare ImageNet dataset
- Install Cuda 10.0
- Install TensorRT 5.x
- Inistall Tensorflow
- Install Opencv 4.x
- Download Tensorpack 0.8.9 code to working directory
  Note that later versions might not work,

   1. `$ cd path/to/your_working_directory/`
   2. `$ wget https://github.com/tensorpack/tensorpack/archive/0.8.9.tar.gz`
   3. `$ tar -xvzf 0.8.9.tar.gz`
   4. Add line to ~/.bashrc

      export PYTHONPATH=$PYTHONPATH:/path/to/tensorpack-0.8.9

   5. `$ source ~/.bashrc`

## Preparing .wts file for TensorRT Inference ##
  1. Clone this repository
  2. `$ cd senet/Revised_Scripts`
  3. Download the `ImageNet-ResNet50-SE.npz` file from the ResNet50-SE model by running the following command:

          wget http://models.tensorpack.com/ResNet/ImageNet-ResNet50-SE.npz

  4. You can find the revised `base_revised.py` script in the `senet/Revised_Scrpits` directory. Then replace the content of the original `base.py` in `tensorpack/tensorpack/predict/`.

     * Note (The difference between base_revised.py and base.py) :

         In order to obtain the Checkpoints and Tensorboard from the Tensorpack (TensorFlow re-implementation),
         we added an tf.train.Saver() object and an tf.summary.FileWriter() object in the OfflinePredictor class
         in the `tensorpack/tensorpack/predict/base.py`

  5. We can use the sample script from tensorpack to obtain checkpoints for senet.

     The checkpoints will be stored in the `SE-ResNet50-ckpt` folder under the same directory (senet/Revised_Scrpits)

      `$ python /path/to/tensorpack-0.8.9/tensorpack/examples/ResNet/imagenet-resnet.py --data [/path/to/ImageNet_dataset] --load [/path/to/ResNet50-SE.npz] -d 50 --eval --mode se`

  6. You can find the revised `dumpTFWts_revised.py` script in the `Revised_Scripts` directory.

      `$ cd SE-ResNet50-ckpt`   
      `$ python ../dumpTFWts_revised.py -m se-resnet50.ckpt -1 true -o SE-ResNet50 `

     * Note (The difference between dumpTFWts_revised.py and dumpTFWts.py) :

       In order to extract the weights from the generated checkpoints, we use the Python script `dumpTFWts.py` provided in the `tensorrt/samples/common/` directory.

       This project manually loads the V1 format .wts file.
       However, in TensorFlow, convolutional weight tensors are in the order of [filter_height, filter_width, input_depth, output_depth],
       see the [TensorFlow documentation](https://www.tensorflow.org/guide/extend/model_files#weight_formats).
       Similarly, weight tensors of fully-connected layer are in the order of [input_depth, output_depth]
       while TensorRT reads in the order of [output_depth, input_depth].
       Therefore, it is necessary to transpose the weight tensors to the correct order before dumping the weights.

  7. `Move your .wts file to senet/data`

## Update configurations ##

  1. Update the TENSORRT_INSTALL_DIR, OPENCV_INSTALL_DIR, and cuda directory in `Makefile.config` file present in the main directory.

  2. Set the kIMAGE_DATASET_DIR to the path of the ImageNet dataset directory in the `lib/network_config.cpp` file.

  3. [OPTIONAL] Update the paths of the and .wts file and other network parameters in senet/lib/network_config.cpp file if required.

## Building and running the trt-se-resnet50-app ##
  1. he trt-senet-app located at `apps/trt-senet/build` is a standalone app, which does inference on test images listed in the eval.txt file in the ImageNet dataset directory.

      This app has two important parameters--kBATCHSIZE and kPRECISION present in the `network_config.cpp` file that can set the batch size and precision respectively.

      To use different batch size, set the kBATCHSIZE parameter to the desired value. The default value is 1.
      To use the INT8 mode, set the kPRECISION parameter to "kINT8". The default value is "kFLOAT".
   2.
      Run the following command to build/install the trt-senet-app using cmake and execute the app.

        ```
        $ cd apps/trt-senet
        $ mkdir build && cd build
        $ cmake -D OPENCV_ROOT=/path/to/opencv-4.0.1/ -D TRT_SDK_ROOT=/path/to/TensorRT -D CMAKE_BUILD_TYPE=Release ..
        $ make
        $ make && sudo make install
        $ cd ../../../  
        $ trt-senet-app       
        ```

  3. After running the app successfully, TensorRT engine file will be stored under `senet/data` by default.
  We can use this TensorRT engine file to do inference using Deepstream later.

## Inference Performance ##

| Model                  | Top1 err.| Top5 err.|
|:-----------------------|:--------:|:--------:|
| Original               |23.29%    |6.62%     |
| TF-re-implementation   |22.64%    |6.24%     |
| TRT-SE-ResNet50-FP32   |22.64%    |6.24%     |
| TRT-SE-ResNet50-INT8   |22.64%    |6.24%     |

Table1: Single-Crop error rates (%) on the ImageNet validation set. The original row refers to the results from the original Squeeze-and-Excitation network papers. The re-implementation row refers to the TensorFlow re-implementation in the above link.


| TRT-SE-ResNet50  |               |TF-re-implementation|               |   |               |TRT-SE-ResNet50-FP32           |               |   |               |TRT-SE-ResNet50-INT8           |               |
|:----------------:|:-------------:|:-------------:|:-------------:|:-:|:-------------:|:-------------:|:-------------:|:-:|:-------------:|:-------------:|:-------------:|
|Batch Size        |Perf (ms/batch)|Perf (ms/frame)|Memory (MiB)   |   |Perf (ms/batch)|Perf (ms/frame)|Memory (MiB)   |   |Perf (ms/batch)|Perf (ms/frame)|Memory (MiB)   |
|1                 |1253.1533	   |1253.1533	   |8515           |   |4.5759         |4.5759         |479            |   |3.2389         |3.2389         |381            |
|2                 |1267.523	   |633.7615	   |8515           |   |6.7534         |3.3767         |511            |   |4.4621         |2.2310         |383            |
|4                 |1317.9121	   |329.478	       |8515           |   |10.1916        |2.5479         |521            |   |6.1109         |1.5277         |395            |
|8                 |1483.5896	   |185.4487       |8515           |   |17.1475        |2.1434         |579            |   |9.4456         |1.1807         |439            |
|16                |1433.175	   |89.5734        |8515           |   |30.5326        |1.9082         |645            |   |15.2030        |0.9501         |425            |
|32                |1632.723	   |51.0225        |8515           |   |57.6064        |1.8002         |829            |   |26.8220        |0.8381         |475            |
|64                |1830.8115	   |28.6064        |8515           |   |104.2368       |1.6287         |1201           |   |50.5845        |0.7903         |537            |
|128               |2218.8096	   |17.3344        |8515           |   |216.7552       |1.6934         |1913           |   |92.9920        |0.7265         |793            |
|256               |3318.3481	   |12.9622        |8515           |   |410.4115       |1.6031         |3151           |   |199.4728       |0.7791         |1401           |
|512               |4901.4744†	   |9.5731†        |8515           |   |796.2572       |1.5551         |5863           |   |423.8336       |0.8278         |3411           |

Table2: Inference speed and memory usage with different batch size comparisons. The TensorFlow inference is running on tensorFlow-gpu 1.12.0 and the TensorRT inference is running on TensorRT-5.0.2.6. † indicates the performance is limited due to running out of memory.

| # of Calib Img   | Top1 err.| Top5 err.|
|:-----------------|:--------:|:--------:|
| 5K               |30.71%    |11.10%    |
| 10K              |30.64%    |11.03%    |
| 15K              |31.12%    |11.39%    |
| 20K              |22.97%    |6.44%     |
| 25K              |22.99%    |6.46%     |
| 30K              |23.01%    |6.48%     |
| 35K              |22.96%    |6.44%     |
| 40K              |23.24%    |6.48%     |
| 50K              |22.99%    |6.46%     |
| 60K              |23.01%    |6.43%     |
| 70K              |22.99%    |6.45%     |

Table3: Accuracy change with the number of calibration image used.

## Note ##

1. Whenever you adjust the network definition or using a different calibration table, it is necessary to delete the .engine files from the `senet/data` directory and generate a new engine in the next run. To delete the .engine files, simply go to data folder and delete the engine file.

2. You can also delete the calibration table by going to data folder and delete the table file if needed.

3. If you run make the app again,
    ```
    $ cd apps/trt-senet
    $ rm -rf build
    ```
