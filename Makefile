# PSCKangaroo v57.0.0 Makefile
# Modular build: .cu files compiled separately, .cpp files included via headers

CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Architecture (adjust for your GPU)
# sm_86 = RTX 30xx/40xx, sm_89 = RTX 4070/5070, sm_90 = H100
ARCH = -arch=sm_89

# Base flags
NVCCFLAGS = -O3 $(ARCH) -maxrregcount=128 --use_fast_math -DNDEBUG
NVCCFLAGS += -I. -Xcompiler -Wall -Xcompiler -Wextra

# Optional features (uncomment to enable)
# TENSOR_CORES = -DUSE_TENSOR_CORES
# DYN_PARALLEL = -rdc=true -lcudadevrt

LDFLAGS = -lcudart -lm

# All source files (kernel.cu contains device code, .cpp contain host code)
# Since .cpp files include each other via common.h, we compile them as CUDA
SOURCES = kernel.cu main.cpp benchmark.cpp checkpoint.cpp dp_table.cpp precompute.cpp verify.cpp

# Target
TARGET = psc_v57

.PHONY: all clean tensor dyn

all: $(TARGET)

tensor: NVCCFLAGS += -DUSE_TENSOR_CORES
tensor: $(TARGET)

dyn: NVCCFLAGS += -rdc=true
dyn: LDFLAGS += -lcudadevrt
dyn: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o $(CHECKPOINT_FILE)

# WSL2 specific notes
wsl-note:
	@echo "WSL2 Notes:"
	@echo "  - Install: sudo apt install nvidia-cuda-toolkit"
	@echo "  - Check GPU: nvidia-smi"
	@echo "  - No TDR timeout in WSL, but kernel launch timeout still possible"
	@echo "  - Reduce STEPS_PER_KERNEL in common.h if cudaErrorLaunchTimeout"
