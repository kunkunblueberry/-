
# EXECUTABLE := cudaSaxpy

# CU_FILES   := saxpy.cu

# CU_DEPS    :=

# CC_FILES   := main.cpp

# ###########################################################

# ARCH=$(shell uname | sed -e 's/-.*//g')

# OBJDIR=objs
# CXX=g++ -m64
# CXXFLAGS=-O3 -Wall
# LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
# NVCC=nvcc
# NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

# OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


# .PHONY: dirs clean

# all: $(EXECUTABLE)

# default: $(EXECUTABLE)

# dirs:
# 		mkdir -p $(OBJDIR)/

# clean:
# 		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

# $(EXECUTABLE): dirs $(OBJS)
# 		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# $(OBJDIR)/%.o: %.cpp
# 		$(CXX) $< $(CXXFLAGS) -c -o $@

# $(OBJDIR)/%.o: %.cu
# 		$(NVCC) $< $(NVCCFLAGS) -c -o $@



EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
# 修改：更新为 CUDA 12.9 库路径（根据实际安装路径调整）
LDFLAGS=-L/usr/local/cuda-12.9/lib64/ -lcudart -ldl

NVCC=nvcc
# 修改：适配 RTX 4050 的 Ada 架构，并抑制警告
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_89 -ccbin /usr/bin/gcc \
          -Wno-deprecated-gpu-targets


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
    