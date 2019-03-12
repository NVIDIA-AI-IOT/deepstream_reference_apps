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
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(deepstream-anomaly-app)
enable_language(C)

find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4
                               gstreamer-video-1.0>=1.4
                               gstreamer-app-1.0>=1.4)

set(DS_SDK_ROOT "" CACHE PATH "NVIDIA Deepstream SDK root path")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

link_directories(${GST_LIBRARY_DIRS} /usr/local/deepstream)

include_directories(${GST_INCLUDE_DIRS} ${DS_SDK_ROOT}/sources/includes)
add_executable(deepstream-anomaly-app deepstream-anomaly-app.c)
target_link_libraries(deepstream-anomaly-app ${GST_LIBRARIES} rt m nvdsgst_helper nvdsgst_meta)

#Install app
install(TARGETS deepstream-anomaly-app RUNTIME DESTINATION bin CONFIGURATIONS Release Debug)
