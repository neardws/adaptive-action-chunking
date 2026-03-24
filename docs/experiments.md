# Experiments Plan

## Exp 1: Latency Characterization ✅
**Goal**: Measure per-k latency on server + edge hardware.

| Model | Hardware | Status |
|-------|----------|--------|
| BitVLA 2.82B | RTX 4090D | ✅ Done |
| BitVLA 2.82B | Jetson Orin NX | ✅ Done |
| π₀-FAST base | RTX 4090D | ✅ Done |
| π₀-FAST INT4 | Jetson Orin NX | ⏳ Needs openvla.cpp |

**Result**: π₀-FAST latency is linear in k (106ms/step @ k=1, +200ms per 3 steps).

## Exp 2: Fixed-k Baseline on LIBERO ❌
**Goal**: Establish SR vs latency Pareto frontier for fixed k.

Tasks: libero_spatial (10 tasks), libero_object (10 tasks)
Episodes per task: 50
k values: 1, 2, 4, 8, 16

**Expected**: SR degrades for very small k (jerky), flat for k=4~8.

## Exp 3: Oracle k Upper Bound ❌
**Goal**: What SR could a perfect k-selector achieve?

Run fixed k=1 episodes, label with oracle k, compute "oracle SR".
This is the upper bound our k-selector should approach.

## Exp 4: k-Selector Training & Eval ❌
**Goal**: Train lightweight MLP k-selector, evaluate on held-out episodes.

Training: 100 oracle-labeled episodes per task
Eval: 50 fresh episodes per task
Compare SR + latency vs fixed-k baselines.

## Exp 5: Ablations ❌
- Feature ablation: visual only / state only / combined
- k candidate set size: {1,4} vs {1,4,8} vs {1,2,4,8,16}
- Variance threshold sensitivity

## Exp 6: Edge Deployment ❌
- π₀-FAST INT4 on Jetson + adaptive k
- Measure SR on physical robot or Jetson-in-the-loop simulation
