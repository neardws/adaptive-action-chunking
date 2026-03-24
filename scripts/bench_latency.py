"""
Latency benchmark for VLA models with variable k.
Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_latency.py --model pi0fast --k 1 4 8 16
"""

import sys, argparse, json, time, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pi0fast", choices=["pi0fast", "bitvla"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="experiments/results/bench_latency.json")
    args = parser.parse_args()

    from src.policy.pi0fast_wrapper import Pi0FastWrapper
    wrapper = Pi0FastWrapper(device=args.device).load()

    # Dummy inputs
    H = W = 224
    images = {
        "observation.images.base_0_rgb": torch.rand(1, 3, H, W),
        "observation.images.left_wrist_0_rgb": torch.rand(1, 3, H, W),
        "observation.images.right_wrist_0_rgb": torch.rand(1, 3, H, W),
    }
    state = torch.zeros(1, 32)
    task = "pick up the object"

    results = []
    print(f"\n{'k':>4}  {'mean_ms':>8}  {'std_ms':>7}  {'Hz':>6}")
    print("-" * 35)

    for k in args.k:
        latencies = []
        for i in range(args.n_warmup + args.n_runs):
            _, lat = wrapper.infer(images, state, task, k)
            if i >= args.n_warmup:
                latencies.append(lat)

        r = {
            "k": k,
            "mean_ms": round(np.mean(latencies), 1),
            "std_ms": round(np.std(latencies), 1),
            "hz": round(1000 / np.mean(latencies), 2),
        }
        results.append(r)
        print(f"{k:>4}  {r['mean_ms']:>8.1f}  {r['std_ms']:>7.1f}  {r['hz']:>6.2f}")

    out = {
        "model": args.model,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
