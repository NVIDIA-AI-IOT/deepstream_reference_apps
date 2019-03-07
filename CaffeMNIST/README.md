CaffeMNIST is another sample which demostrates inference in deepstream using the NvInfer plugin with custom networks consisting of plugin layers. The example network is the standard LeNet which has been modified to recieve RGB images. We perform inference on video streams generated from the mnist test image dataset.

Pre-requisites:
- Copy mnist_mean.binaryproto and mnist.caffemodel from the `data/mnist` directory in the TensorRT SDK to the `CaffeMNIST/data` directory

- Using the mnist test images, create sample videos for each digit from 0-9 using ffmpeg as follows.

  1) Install ffmpeg : `$ sudo apt-get install ffmpeg`
  
  2) For each digit from 0-9, run the following command to generate videos from their corresponding test images
     `ffmpeg -framerate 10 -pattern_type glob -i '*.png' -vf negate,scale=1280:720 -c:v libx264 -pix_fmt yuv420p ~/.720p-0.mp4`   
     This command generates a video file with name `720p-0.mp4` in the current folder by combining all the test images corresponding to digit 0.
  
  3) Move all the video streams, one for each digit into the `CaffeMNIST/data` directory

We use cmake to compile the library:   
Set the DS_SDK_ROOT variable to point to your DeepStream SDK root directory and TRT_SDK_ROOT to TensorRT SDK root directory.

   `$ cd nvdsinfer_custom_impl_CaffeMNIST` 
   `$ mkdir build && cd build`   
   `$ cmake -D DS_SDK_ROOT=<DS SDK Root> -D TRT_SDK_ROOT=<TRT SDK Root> -D CMAKE_BUILD_TYPE=Release ..`   
   `$ make`
   `$ cd ../../`

Run the sample:
  ` $deepstream-app -c deepstream_app_config_CaffeMNIST.txt`