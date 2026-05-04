# Build Requirements

## CUDA 13.2 + GCC 16 = Broken

GCC 16.1.1 introduced `__builtin_is_virtual_base_of` and `__self` deduction guides in libstdc++ headers. NVCC 13.2 can't parse these. Affects all CUDA projects (llama.cpp, lucebox, megakernel, PyTorch extensions).

### Fix: Always use gcc-14

```bash
sudo pacman -S gcc14  # Arch/CachyOS
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
```

### Per-project commands

**llama.cpp:**
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
```

**Megakernel (torch extension):**
```bash
MEGAKERNEL_CUDA_ARCH=sm_89 CXX=g++-14 CUDAHOSTCXX=g++-14 \
  python setup.py build_ext --inplace
```

**Lucebox DFlash:**
```bash
CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
```

## Clang Native CUDA

Clang 22 can compile CUDA natively (no NVCC) but:
- CUDA 13.2 CCCL headers use `__host__`-only deduction guides → clang error
- Supported up to CUDA 12.9
- CMakeLists patch for flag stripping written but untested end-to-end
- Will work when either CUDA or clang updates

## Persistent Setup

Add to `~/.zshrc`:
```bash
export CUDAHOSTCXX=g++-14
```
