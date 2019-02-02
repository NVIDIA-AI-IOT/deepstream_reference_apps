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

SRCS:= gstyoloplugin.cpp
INCS:= $(wildcard *.h)
LIB:=libgstnvyolo.so

CONFIG :=../../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

ifeq ($(CUDA_VER),)
$(error "CUDA_VER variable is not set in Makefile.config")
endif

ifeq ($(PLATFORM),)
$(error "PLATFORM variable is not set in Makefile.config")
endif

ifneq ($(PLATFORM), TEGRA)
$(error "This plugin is developed for TEGRA platform. Check PLATFORM variable in Makefile.config")
endif

ifeq (' $(DEEPSTREAM_INSTALL_DIR)', ' /path/to/DeepStream_Release_3.0')
$(error DEEPSTREAM_INSTALL_DIR variable is not set in Makefile.config)
endif

SUBDIR :=../../lib
DEP:=$(SUBDIR)/libyoloplugin.a
DEP_FILES:=$(wildcard $(SUBDIR)/*.cpp)
DEP_FILES+=$(wildcard $(SUBDIR)/*.h)

CXXFLAGS := -O2 -fPIC -std=c++11 -lstdc++fs -Wall -Wunused-function -Wunused-variable -Wfatal-errors
INCS:= -I "/usr/local/cuda-$(CUDA_VER)/include" \
	   -I "$(DEEPSTREAM_INSTALL_DIR)/sources/includes" \
       -I "/usr/include/aarch64-linux-gnu" \
	   -I "$(SUBDIR)/"

LIBS := -shared -Wl,-no-undefined \
  -L/usr/lib/aarch64-linux-gnu/tegra -lnvbuf_utils -lgstnvquery -Wl,-rpath="/usr/lib/aarch64-linux-gnu/tegra" \
  -L/usr/lib/aarch64-linux-gnu -lnvinfer -lnvinfer_plugin -lgstnvdsmeta -lgflags -lEGL -Wl,-rpath="/usr/lib/aarch64-linux-gnu" \
  -L $(SUBDIR) -lyoloplugin -Wl,-rpath=$(SUBDIR) \
  -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnppc -lnppig -lnpps -Wl,-rpath="/usr/local/cuda-$(CUDA_VER)/lib64"

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv
CXXFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

.PHONY: all install clean clean_models reinstall

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(INCS) $(CXXFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo "Building $(LIB)"
	$(CXX) -o $@ $(OBJS) $(DEP) $(LIBS) $(CXXFLAGS) $(INCS)

$(DEP): $(DEP_FILES)
	@echo "Building lib"
	$(MAKE) -C $(SUBDIR)

install: $(LIB)
	@echo "Installing $(LIB)"
	cp -rv $(LIB) /usr/lib/aarch64-linux-gnu/gstreamer-1.0/

clean:
	rm -rf $(OBJS) $(LIB)
	$(MAKE) -C $(SUBDIR) clean

clean_models:
	$(MAKE) -C $(SUBDIR) clean_models

reinstall: $(LIB)
	@echo "Re installing $(LIB)"
	$(MAKE) clean
	$(MAKE) all
	$(MAKE) install