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


APP:= deepstream-yolo-app

SRCS:= $(wildcard *.cpp)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.cpp=.o)

LIB:= ../../lib
CONFIG :=../../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

CXXFLAGS:= -I $(DEEPSTREAM_INSTALL_DIR)/sources/includes \
		   -I /usr/include/gstreamer-1.0	

CXXFLAGS+= -std=c++11 -Wunused-variable -Wfatal-errors `pkg-config --cflags $(PKGS)`

LIBS:= `pkg-config --libs $(PKGS)`

.PHONY: all install clean clean_models reinstall

all: $(APP)

.o: .cpp $(INCS) Makefile
	$(CXX) -c $@ $(CXXFLAGS) $<

$(APP): $(OBJS) Makefile
	@echo Building $(APP)
	$(CXX) -o $(APP) $(OBJS) $(CXXFLAGS) $(LIBS) 

install: $(APP)
	@echo Installing $(APP)
	cp -rv $(APP) /usr/bin/

clean:
	rm -rf $(OBJS) $(APP)
clean_models:
	$(MAKE) -C $(LIB) clean_models
reinstall: $(APP)
	@echo "Re installing $(APP)"
	$(MAKE) clean
	$(MAKE) all
	$(MAKE) install
