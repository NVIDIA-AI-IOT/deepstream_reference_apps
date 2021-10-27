################################################################################
# Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

CXX:= g++
SRCS:= gstdsdirection.cpp
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_dsdirection.so

DS_SDK_ROOT:=/opt/nvidia/deepstream/deepstream

DEP:=dsdirection_lib/libdsdirection.a
DEP_FILES:=$(wildcard dsdirection_lib/dsdirection_lib.* )
DEP_FILES-=$(DEP)

CFLAGS+= -fPIC -DDS_VERSION=\"6.0.0\" \
	 -I $(DS_SDK_ROOT)/sources/includes

GST_INSTALL_DIR?=$(DS_SDK_ROOT)/lib/gst-plugins/
LIB_INSTALL_DIR?=$(DS_SDK_ROOT)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L dsdirection_lib -ldsdirection \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS) $(DEP)

$(DEP): $(DEP_FILES)
	$(MAKE) -C dsdirection_lib/

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
