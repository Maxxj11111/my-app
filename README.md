# PSCKangaroo v57.0.0 — Puzzle 135 Solver

## Architecture
Modular CUDA/C++ project with separation of concerns:

```
psckangaroo_v57/
├── common.h          — Constants, types, inline device/host math
├── mont_math.cuh     — Montgomery domain arithmetic (device + host)
├── glv.cuh           — GLV decomposition + endomorphism
├── ec_ops.cuh        — EC operations (SoA, mixed add, batch invert)
├── tensor_core.cuh   — WMMA Tensor Cores (optional, sm_80+)
├── hash_dp.cuh       — Hash functions, DP pack/unpack, collision check
├── kernel.cu         — Main GPU kernel v57 + device constants
├── host_math.cpp     — Host EC math (Pippenger, windowed, batch inv)
├── precompute.cpp    — Jump table, T0, GLV table generation
├── dp_table.cpp      — Multi-GPU DP table + async sync + NVLink
├── checkpoint.cpp    — Save/load checkpoint with merged state
├── verify.cpp        — Key verification (4 possibilities)
├── benchmark.cpp     — Multi-GPU benchmark mode
├── main.cpp          — CLI, GPU setup, main search loop
└── Makefile          — Build system with optional features
```

## Features
- **Montgomery arithmetic**: O(1) reduction, no divisions
- **GLV decomposition**: k = k1 + k2*lambda, ~1.7x speedup
- **Pippenger algorithm**: Batch multi-scalar multiplication
- **Batch inversion**: Montgomery trick O(n) vs O(n*log p)
- **SoA layout**: Structure of Arrays for better coalescing
- **Cooperative Groups**: Warp-level ballot/reduce/shfl
- **Multi-GPU**: Pinned memory sync, NVLink zero-copy auto-detect
- **Dynamic Parallelism**: Adaptive kernel launch (optional)
- **Tensor Cores**: WMMA for 256-bit multiply (optional, sm_80+)

## Compilation

```bash
# Standard build (RTX 4070/5070)
make

# With Tensor Cores (RTX 3090/4090/5090, sm_86+)
make tensor

# With Dynamic Parallelism
make dyn

# Manual compilation
nvcc -O3 -arch=sm_89 -maxrregcount=128 --use_fast_math -DNDEBUG \
     -o psc_v57 kernel.cu main.cpp host_math.cpp precompute.cpp \
     dp_table.cpp checkpoint.cpp verify.cpp benchmark.cpp -lcudart
```

## Usage

```bash
# Default mode (Montgomery + GLV + Pippenger)
./psc_v57 <pubkey_x> <pubkey_y>

# With mod3 filter
./psc_v57 <pubkey_x> <pubkey_y> --mod3

# Benchmark
./psc_v57 --benchmark --gpus 2

# Resume from checkpoint
./psc_v57 <pubkey_x> <pubkey_y> --resume

# Multi-GPU
./psc_v57 <pubkey_x> <pubkey_y> --gpus 4
```

## Performance Estimates (RTX 5070)
| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Base (v53) | 1x | 1x |
| 64-bit limbs | 4x | 4x |
| Montgomery | 2x | 8x |
| Negation map | 1.4x | 11x |
| mod3 filter | 1.7x | 19x |
| GLV | 1.7x | 32x |
| SoA + CG | 1.3x | 42x |
| Multi-GPU (2x) | 2x | 84x |

## Known Issues
1. GLV decomposition on device is simplified (full 256-bit division needed)
2. Tensor Cores integration is partial (falls back to standard mul)
3. Pippenger in precompute uses windowed fallback (needs fix)

## TODO
- [ ] Full 256-bit GLV decomposition on device
- [ ] Complete Tensor Core matrix layout
- [ ] SIMD-optimized Pippenger bucket sum
- [ ] Asynchronous DP sync with events
- [ ] Dynamic grid sizing based on occupancy
