# Adaptive Action Chunking — Design Document

## Problem Statement

VLA models using FAST tokenization generate action chunks autoregressively.
Latency scales linearly with chunk size k:

| k | Latency (π₀-FAST, RTX 4090D) | Use Case |
|---|-------------------------------|----------|
| 1 | 106ms (9.4Hz) | Precision tasks, recovery |
| 4 | 306ms (3.3Hz) | Normal manipulation |
| 8 | 574ms (1.7Hz) | Smooth trajectories |
| 16 | 1116ms (0.9Hz) | Static reaching |

**Fixed k** forces a trade-off: either low latency (small k, many VLA calls)
or smooth motion (large k, slow reaction). **Adaptive k** eliminates this.

## k-Selector Design

### Input Features

1. **Visual**: SigLIP pooled features [1152-d], extracted from π₀-FAST's frozen vision tower
2. **State**: Robot joint positions/velocities [32-d]

### Output

Discrete class ∈ {1, 2, 4, 8, 16}, converted to k value.

### Architecture

```
[SigLIP features, 1152] ─┐
                           ├─▶ Linear(1184→128) ─▶ ReLU ─▶ 
[robot state, 32]         ─┘                              
                              Linear(128→64) ─▶ ReLU ─▶ Linear(64→5)
                                                           ↓
                                                       softmax → k
```

Parameters: ~180K (negligible overhead)
Inference: <1ms (not on critical path)

### Training Data: Oracle Labels

1. Collect N episodes using fixed k=1
2. For each timestep t, compute oracle k:
   - Look ahead at future actions [t:t+k_max]  
   - Find largest k where action variance < threshold
3. Train with cross-entropy loss

### Training Signal Intuition

- High action variance → scene is dynamic, use small k
- Low action variance → smooth trajectory, use large k
- This captures "how predictable is the near future"

## Evaluation Protocol

### Metrics

| Metric | Definition |
|--------|-----------|
| SR (Success Rate) | % episodes completed |
| Mean Latency | avg VLA call time per episode |
| Effective Hz | episode_steps / total_time |
| Mean k | avg chunk size used |

### Baselines

1. Fixed k=1 (lower bound latency, upper bound reactivity)
2. Fixed k=4 (current standard)
3. Fixed k=8
4. Fixed k=16 (upper bound smoothness)
5. **Adaptive k (ours)**

### Expected Results

Adaptive k should achieve:
- SR ≈ fixed k=4 baseline (no degradation)  
- Mean latency ≈ fixed k=8 or better (lower overhead)
- Effective Hz between k=4 and k=8

## Edge Deployment Note

For Jetson Orin NX:
- INT4 GGUF quantization (paligemma_q8_0.gguf already downloaded)
- openvla.cpp or equivalent C++ inference
- Expected k=4 latency: ~200-400ms (target: ≥5Hz)
