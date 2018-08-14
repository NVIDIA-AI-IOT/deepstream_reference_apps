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

CUDA_VER:=9.2
CXX:= g++
SRCS:= gstyoloplugin.cpp
INCS:= $(wildcard *.h)
LIB:=libgstnvyolo.so

CONFIG :=../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

SUBDIR := yoloplugin_lib
DEP:=$(SUBDIR)/libyoloplugin.a
DEP_FILES:=$(wildcard $(SUBDIR)/*.cpp)
DEP_FILES+=$(wildcard $(SUBDIR)/*.h)

CXXFLAGS := -fPIC -std=c++11 -lstdc++fs -Wall -Wunused-function -Wunused-variable
LIBS := -shared -Wl,-no-undefined \
  -L $(SUBDIR) -lyoloplugin \
  -L$(OPENCV_INSTALL_DIR)/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_imgcodecs -Wl,-rpath="$(OPENCV_INSTALL_DIR)/lib"\
  -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnppc -lnppig -lnpps -Wl,-rpath="/usr/local/cuda-$(CUDA_VER)/lib64"\
  -L/usr/local/deepstream/ -lgstnvquery -lgstnvdsmeta -Wl,-rpath,/usr/local/deepstream/ \
  -L $(TENSORRT_INSTALL_DIR)/lib -lnvinfer -lnvinfer_plugin -Wl,-rpath="$(TENSORRT_INSTALL_DIR)/lib"


CXXFLAGS+= \
	-I /usr/local/cuda-$(CUDA_VER)/include \
	-I $(DEEPSTREAM_INSTALL_DIR)/sources/includes \
    -I $(OPENCV_INSTALL_DIR)/include \
	-I $(TENSORRT_INSTALL_DIR)/include


OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CXXFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CXXFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS) $(CXXFLAGS)

$(DEP): $(DEP_FILES)
	@echo "Building yoloplugin_lib"
	$(MAKE) -C $(SUBDIR) CXX="$(CXX)" CUDA_VER="$(CUDA_VER)" TENSORRT_INSTALL_DIR="$(TENSORRT_INSTALL_DIR)"

install: $(LIB)
	cp -rv $(LIB) /usr/lib/x86_64-linux-gnu/gstreamer-1.0/

clean:
	rm -rf $(OBJS) $(LIB)
	$(MAKE) -C $(SUBDIR) clean
