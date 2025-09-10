################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-ipc-test-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

SRCS:= $(wildcard *.c)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I /usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser -lgstapp-1.0 \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS)
	$(CC) -o $(APP) $(OBJS) $(LIBS)
	make -C video_template_impl
	make -C latency_serialization
	@echo "Build complete."

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)
	make -C video_template_impl install
	make -C latency_serialization install
	@echo "Install complete."

clean:
	make -C video_template_impl clean
	make -C latency_serialization clean
	rm -rf $(OBJS) $(APP)
	@echo "Clean complete."

.PHONY: all
