"""
Generate oracle k labels for k-Selector training.

Strategy: Run episode with k=1 (finest granularity).
For each timestep t, look ahead at the next N steps:
- If actions are smooth and consistent → large k is OK
- If action variance is high → small k needed

This gives a "hindsight optimal k" label for each observation.
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path


def compute_oracle_k(
    actions_future: np.ndarray,
    k_candidates: list[int] = [1, 2, 4, 8, 16],
    variance_threshold: float = 0.1,
) -> int:
    """
    Given future actions (T, action_dim), compute the oracle k.
    Largest k such that action variance within the chunk is below threshold.
    """
    for k in reversed(k_candidates):
        if len(actions_future) < k:
            continue
        chunk = actions_future[:k]
        var = np.var(chunk, axis=0).mean()
        if var < variance_threshold:
            return k
    return k_candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/oracle_labels.jsonl")
    parser.add_argument("--variance_threshold", type=float, default=0.05)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes_dir = Path(args.episodes_dir)
    all_labels = []

    for ep_file in sorted(episodes_dir.glob("*.npz")):
        data = np.load(ep_file)
        actions = data["actions"]  # [T, action_dim]
        T = len(actions)

        ep_labels = []
        for t in range(T):
            future = actions[t:]
            k_oracle = compute_oracle_k(future, variance_threshold=args.variance_threshold)
            ep_labels.append(k_oracle)

        all_labels.append({
            "episode": ep_file.stem,
            "k_labels": ep_labels,
            "mean_k": float(np.mean(ep_labels)),
        })
        print(f"  {ep_file.stem}: mean_k={np.mean(ep_labels):.1f}")

    with open(output_path, "w") as f:
        for item in all_labels:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(all_labels)} episodes to {output_path}")


if __name__ == "__main__":
    main()
