# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CC:= gcc

SRCS:= latency_serialization.c
LIB:=liblatency_serialization.so

CFLAGS+= -fPIC

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

LIBS += -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.c=.o)
PKGS:= gstreamer-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.c $(INCS)
	@echo $(CFLAGS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS)
	@echo $(CFLAGS)
	$(CC) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
