"""
LIBERO benchmark evaluator for fixed-k vs adaptive-k comparison.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
from pathlib import Path


@dataclass
class EpisodeResult:
    task: str
    k_sequence: list[int]
    success: bool
    total_latency_ms: float
    n_vla_calls: int
    mean_latency_ms: float = 0.0

    def __post_init__(self):
        self.mean_latency_ms = self.total_latency_ms / max(self.n_vla_calls, 1)


@dataclass
class EvalConfig:
    tasks: list[str] = field(default_factory=lambda: ["libero_spatial"])
    n_episodes: int = 50
    max_steps: int = 600
    fixed_k_baselines: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    use_adaptive: bool = True
    device: str = "cuda:0"
    result_dir: str = "experiments/results"


class LiberoEvaluator:
    """
    Runs LIBERO tasks and measures:
    - Task Success Rate (SR)
    - Mean latency per VLA call
    - P90 latency
    - Effective Hz (steps / total_time)
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.result_dir = Path(config.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def run_fixed_k(self, policy_wrapper, env, k: int, n_episodes: int) -> dict:
        """Run episodes with fixed chunk size k."""
        results = []
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            step = 0
            total_latency = 0.0
            n_calls = 0

            while not done and step < self.config.max_steps:
                actions, latency = policy_wrapper.infer(
                    images=obs["images"],
                    state=obs["state"],
                    task=obs["task"],
                    k=k,
                )
                total_latency += latency
                n_calls += 1

                for i in range(k):
                    if done:
                        break
                    obs, reward, done, info = env.step(actions[i].cpu().numpy())
                    step += 1

            results.append(EpisodeResult(
                task=obs.get("task", "unknown"),
                k_sequence=[k] * n_calls,
                success=info.get("success", False),
                total_latency_ms=total_latency,
                n_vla_calls=n_calls,
            ))

        return self._summarize(results, label=f"fixed_k={k}")

    def run_adaptive(self, policy_wrapper, k_selector, env, n_episodes: int) -> dict:
        """Run episodes with adaptive k-selector."""
        results = []
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            step = 0
            total_latency = 0.0
            n_calls = 0
            k_seq = []

            while not done and step < self.config.max_steps:
                # Get SigLIP features for k prediction
                features = policy_wrapper.get_siglip_features(obs["images"])
                state_t = obs["state"].unsqueeze(0)
                k = k_selector.predict_k(features, state_t)[0]
                k_seq.append(k)

                actions, latency = policy_wrapper.infer(
                    images=obs["images"],
                    state=obs["state"],
                    task=obs["task"],
                    k=k,
                )
                total_latency += latency
                n_calls += 1

                for i in range(k):
                    if done:
                        break
                    obs, reward, done, info = env.step(actions[i].cpu().numpy())
                    step += 1

            results.append(EpisodeResult(
                task=obs.get("task", "unknown"),
                k_sequence=k_seq,
                success=info.get("success", False),
                total_latency_ms=total_latency,
                n_vla_calls=n_calls,
            ))

        return self._summarize(results, label="adaptive_k")

    def _summarize(self, results: list[EpisodeResult], label: str) -> dict:
        latencies = [r.mean_latency_ms for r in results]
        k_dist = {}
        for r in results:
            for k in r.k_sequence:
                k_dist[k] = k_dist.get(k, 0) + 1
        total_k = sum(k_dist.values())
        summary = {
            "label": label,
            "n_episodes": len(results),
            "success_rate": sum(r.success for r in results) / len(results),
            "mean_latency_ms": float(np.mean(latencies)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "mean_k": float(np.mean([k for r in results for k in r.k_sequence])),
            "k_distribution": {k: v / total_k for k, v in k_dist.items()},
        }
        print(f"[{label}] SR={summary['success_rate']:.2%}  "
              f"lat={summary['mean_latency_ms']:.0f}ms  "
              f"mean_k={summary['mean_k']:.1f}")
        return summary

    def save_results(self, all_results: list[dict], filename: str = "eval_results.json"):
        path = self.result_dir / filename
        path.write_text(json.dumps(all_results, indent=2))
        print(f"Results saved to {path}")
