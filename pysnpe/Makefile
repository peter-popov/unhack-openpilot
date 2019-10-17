# Copyright (c) 2017 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

# Specify the compiler flags
#CXX ?= g++

CXXFLAGS += -std=c++11 -fPIC -march=x86-64

SNPE_ROOT =./snpe-1.30.0.480

# Include paths
INCLUDES += -I $(SNPE_ROOT)/include/zdl -I /opt/venv/include/python3.6

# Specify the paths to the libraries
LDFLAGS  += -shared -L $(SNPE_ROOT)/lib/x86_64-linux-clang


# Specify the link libraries
LLIBS    += -lSNPE
LLIBS    += -lboost_python3
LLIBS    += $(SNPE_ROOT)/lib/x86_64-linux-clang/libsymphony-cpu.so


# Specify the target
PROGRAM  := pysnpe
SRC_DIR  := src
OBJ_DIR  := obj

# Specify the source list and the object directories.
# The source file is in the 'jni' directory to be compatible with the
# Android NDK structure.
# Get the list of source files including the directory
SRC      := $(wildcard $(SRC_DIR)/*.cc)
# Generate the output names by substituting the object dir for the source dir
OBJS     := $(subst $(SRC_DIR),$(OBJ_DIR),$(subst .cc,.o,$(SRC)))

default: all
all: $(OBJ_DIR)/$(PROGRAM)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(OBJ_DIR)/$(PROGRAM): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LLIBS) -o $@.so

clean:
	-rm -f $(OBJS) $(PROGRAM).o
	-rm -f $(PROGRAM).so

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: default clean
