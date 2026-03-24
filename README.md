# Adaptive Action Chunking for Edge VLA

> Dynamic k-selection for real-time robot control with π₀-FAST

## Overview

This repository implements **Adaptive Action Chunking** — a lightweight k-selector module that dynamically determines the action chunk size k for Vision-Language-Action (VLA) models, enabling real-time robotic control on edge devices.

## Motivation

| Fixed k | Latency | Hz | Drawback |
|---------|---------|-----|---------|
| k=1 | 106ms | 9.4Hz | Jerky, high VLA call overhead |
| k=4 | 306ms | 3.3Hz | Slow for dynamic scenes |
| k=16 | 1116ms | 0.9Hz | Cannot react to sudden changes |
| **Adaptive k** | ~150ms avg | ~6Hz | ✅ Best of both worlds |

## Architecture

```
Observation ──▶ [k-Selector] ──▶ k_t
     │                              ↓
     └──────▶ [π₀-FAST VLA] ──▶ action_chunk[k_t]
                                    ↓
                           [Action Executor] ──▶ Robot
```

## Modules

- **k-Selector** (`src/selector/`): Lightweight classifier (SigLIP features + MLP)
- **VLA Policy Wrapper** (`src/policy/`): π₀-FAST inference with variable k
- **LIBERO Evaluator** (`src/eval/`): Task success rate + latency benchmarks
- **Experiments** (`experiments/`): Training scripts and ablation configs

## Benchmarks

### π₀-FAST (RTX 4090D)
| k | Latency | Hz |
|---|---------|-----|
| 1 | 106ms | 9.4 |
| 4 | 306ms | 3.3 |
| 8 | 574ms | 1.7 |
| 16 | 1116ms | 0.9 |

### BitVLA (Jetson Orin NX) — Baseline / Counter-example
| k | Latency | Hz |
|---|---------|-----|
| 1 | 1837ms | 0.5 |
| 4 | 6397ms | 0.16 |

## Getting Started

```bash
# Install dependencies
pip install -e ".[pi]" 

# Run latency benchmark
python scripts/bench_latency.py --model pi0fast --k 1 4 8 16

# Train k-selector
python scripts/train_selector.py --config configs/selector_libero.yaml

# Evaluate adaptive k vs fixed k
python scripts/eval_libero.py --mode adaptive --tasks libero_spatial
```

## Citation

```bibtex
@article{xu2026adaptive,
  title={Adaptive Action Chunking for Real-Time Edge VLA Control},
  author={Xu, Xincao},
  year={2026}
}
```

