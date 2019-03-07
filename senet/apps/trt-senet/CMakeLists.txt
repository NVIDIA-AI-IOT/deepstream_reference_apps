# /**
# MIT License

# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *
# */

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(trt-senet-app LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -lstdc++fs -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(TRT_SDK_ROOT "" CACHE PATH "NVIDIA TensorRT SDK root path")
set(OPENCV_ROOT "" CACHE PATH "OpenCV SDK root path")

find_package(CUDA 10.0 EXACT REQUIRED cudart cublas curand)
list(APPEND GPU_ARCHS 30 35 37 50 52 60 61 70 75)

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")


# Find OpenCV
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn PATHS ${OPENCV_ROOT} ${CMAKE_SYSTEM_PREFIX_PATH} PATH_SUFFIXES release installed build share NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn)

set(OpenCV_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  ${OpenCV_INSTALL_PATH}/build/include
  ${OpenCV_INSTALL_PATH}/build/include/opencv4
  ${OpenCV_INSTALL_PATH}/installed/include
  ${OpenCV_INSTALL_PATH}/installed/include/opencv4
)

# Find TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES include samples common)
find_path(TRT_INCLUDE_DIR2 common.h HINTS ${TRT_SDK_ROOT}/samples PATH_SUFFIXES common)

if(${TRT_INCLUDE_DIR} MATCHES "TRT_INCLUDE_DIR-NOTFOUND")
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT headers. Please set path using -DTRT_SDK_ROOT")
else()
  MESSAGE(STATUS "Found TensorRT headers at ${TRT_INCLUDE_DIR}")
  MESSAGE(STATUS "Found TensorRT headers2 at ${TRT_INCLUDE_DIR2}")
endif()

set(TRT_INCLUDE_DIR
  ${TRT_INCLUDE_DIR}
  ${TRT_INCLUDE_DIR2}
)

find_library(TRT_LIBRARY_INFER nvinfer HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
if((${TRT_LIBRARY_INFER} MATCHES "TRT_LIBRARY_INFER-NOTFOUND") OR (${TRT_LIBRARY_INFER_PLUGIN} MATCHES "TRT_LIBRARY_INFER_PLUGIN-NOTFOUND"))
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT libs. Please set path using -DTRT_SDK_ROOT")
else()
  set(TRT_LIBRARY ${TRT_LIBRARY_INFER} ${TRT_LIBRARY_INFER_PLUGIN})
  MESSAGE(STATUS "Found TensorRT libs at ${TRT_LIBRARY}")
endif()

# Add senet lib as subdir
add_subdirectory(${PROJECT_SOURCE_DIR}/../../lib ${PROJECT_BINARY_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/../../lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(trt-senet-app trt-senet-app.cpp)
target_link_libraries(trt-senet-app senet-lib ${OpenCV_LIBS})

#Install app
install(TARGETS trt-senet-app RUNTIME DESTINATION bin CONFIGURATIONS Release Debug)
