# Delta-Net Kernel Development Makefile
# Target: AMD MI50 (gfx906)

HIPCC := /opt/rocm/bin/hipcc
ROCM_PATH := /opt/rocm

# gfx906 specific flags
GPU_ARCH := gfx906
HIPFLAGS := --offload-arch=$(GPU_ARCH)
HIPFLAGS += -O2 -g
HIPFLAGS += -std=c++17
HIPFLAGS += -Wall -Wextra
HIPFLAGS += -I$(ROCM_PATH)/include

# Linker flags
LDFLAGS := -L$(ROCM_PATH)/lib -lhiprtc -lamdhip64

# Environment for gfx906
export HSA_OVERRIDE_GFX_VERSION := 9.0.6
export HIP_VISIBLE_DEVICES := 0

# Source files
SRCS := test_harness.cpp reference.cpp
OBJS := $(SRCS:.cpp=.o)

# Target
TARGET := test_deltanet

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET) for $(GPU_ARCH)"

# Compile .cpp files (including HIP kernels)
%.o: %.cpp reference.h
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

# Run all tests
test: $(TARGET)
	@echo "Running all tests..."
	./$(TARGET)

# Run specific phase
test-phase1: $(TARGET)
	./$(TARGET) --phase 1

test-phase2: $(TARGET)
	./$(TARGET) --phase 2

# Run specific test
test-%: $(TARGET)
	./$(TARGET) --test $*

# List available tests
list: $(TARGET)
	./$(TARGET) --list

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJS) *.bin

# Debug build
debug: HIPFLAGS += -DDEBUG -O0
debug: clean all

# Show GPU info
gpu-info:
	/opt/rocm/bin/rocminfo | grep -E "Name|Marketing|gfx"

# Verify environment
check-env:
	@echo "HIPCC: $(HIPCC)"
	@echo "GPU_ARCH: $(GPU_ARCH)"
	@echo "HSA_OVERRIDE_GFX_VERSION: $(HSA_OVERRIDE_GFX_VERSION)"
	@$(HIPCC) --version

.PHONY: all test test-phase1 test-phase2 list clean debug gpu-info check-env
