#-------------------------------------------------------------------------------
# MIT License
#
# Copyright (C) 2019 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#-------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.0)
project(gstdsopticalflow)
enable_language(C CXX)

set(DS_SDK_ROOT "" CACHE STRING "Root path of NVIDIA Deepstream SDK")
set(OPENCV_ROOT "" CACHE PATH "OpenCV SDK root path")

find_package(CUDA 10.0 REQUIRED)
find_package(PkgConfig)

# Find OpenCV
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn PATHS ${OPENCV_ROOT} ${CMAKE_SYSTEM_PREFIX_PATH} PATH_SUFFIXES build share NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn)

set(OpenCV_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  ${OpenCV_INSTALL_PATH}/build/include
  ${OpenCV_INSTALL_PATH}/build/include/opencv4
  ${OpenCV_INSTALL_PATH}/installed/include
  ${OpenCV_INSTALL_PATH}/installed/include/opencv4
)

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4
                               gstreamer-video-1.0>=1.4
                               gstreamer-app-1.0>=1.4)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(dsopticalflow_lib)

include_directories(${CUDA_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${DS_SDK_ROOT}/sources/includes)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${GST_LIBRARY_DIRS} /usr/local/deepstream)

add_library(gstdsopticalflow SHARED gstdsopticalflow.cpp)

target_link_libraries(gstdsopticalflow cudart nppc nppig npps nppicc nppidei nvdsgst_helper nvdsgst_meta dsopticalflow)
install(TARGETS gstdsopticalflow LIBRARY DESTINATION "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/" CONFIGURATIONS Release Debug)
