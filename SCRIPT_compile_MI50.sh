#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝      
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║        
           ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗                 
          ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝                
          ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗                
          ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗                
          ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝                


EOF

set -e

# 1. Check location
[[ ! -f "CMakeLists.txt" ]] && echo "Error: Not in llama.cpp root directory" && exit 1

# 2. Setup ROCm Environment Variables
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}

# 3. Auto-detect GPU Architecture
# This attempts to find the arch (e.g., gfx1100, gfx1030). 
# If it fails, it defaults to 'native' which tells the compiler to figure it out.
if command -v amdgpu-arch &> /dev/null; then
    AMDGPU_ARCH=$(amdgpu-arch | head -n 1)
    echo "Detected AMD GPU Architecture: $AMDGPU_ARCH"
else
    echo "Warning: amdgpu-arch tool not found. Defaulting to 'native'."
    AMDGPU_ARCH="native"
fi

rm -rf build && mkdir -p build && cd build

# 4. Run CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_HIPBLASLT=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_STATIC=OFF

make -j$(nproc)

echo ""
echo "Build complete: ./build/bin/llama-cli, llama-server, llama-bench"
