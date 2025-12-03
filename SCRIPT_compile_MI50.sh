#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝

    ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
   ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
   ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
   ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
   ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
    ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝

EOF

set -e

[[ ! -f "CMakeLists.txt" ]] && echo "Error: Not in llama.cpp root directory" && exit 1

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}
export HCC_AMDGPU_TARGET=gfx906
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export AMDGPU_TARGETS=gfx906
export GPU_TARGETS=gfx906

rm -rf build && mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -DGGML_HIP_GFX906_OPTIMIZED -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH --offload-arch=gfx906 -DGGML_HIP_GFX906_OPTIMIZED -Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result -mllvm -amdgpu-simplify-libcall -mllvm -amdgpu-internalize-symbols -mllvm -amdgpu-enable-lower-module-lds=false -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -ffast-math -ffp-contract=fast" \
    -DGGML_HIP=ON \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_EXPORT_METRICS=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_NO_PEER_COPY=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_STATIC=OFF

make -j$(nproc)

echo ""
echo "Build complete: ./build/bin/llama-cli, llama-server, llama-bench"
