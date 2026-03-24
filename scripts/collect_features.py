"""
Collect SigLIP features + states from oracle episodes (fixed k=1).
Used to build the training dataset for k-Selector.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/collect_features.py \
    --tasks libero_spatial --n_episodes 100 --output data/features
"""

import sys, os, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="libero_spatial")
    parser.add_argument("--task_ids", nargs="*", type=int, default=None)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--model_id", default="lerobot/pi0fast-libero")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from src.policy.pi0fast_wrapper import Pi0FastWrapper
    from scripts.run_eval import make_env, obs_to_batch

    policy = Pi0FastWrapper(device=args.device, model_id=args.model_id).load()

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.tasks]()
    task_ids = args.task_ids or list(range(suite.get_num_tasks()))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_count = 0

    for tid in task_ids:
        env, task_lang = make_env(args.tasks, tid)
        eps_per_task = max(1, args.n_episodes // len(task_ids))

        for ep_idx in range(eps_per_task):
            obs = env.reset()
            done = False
            step = 0
            ep_features, ep_states, ep_actions = [], [], []

            while not done and step < 600:
                batch = obs_to_batch(obs, task_lang, args.device)
                img_dict = {k: v for k, v in batch.items() if "observation.images" in k}

                # Extract features
                features = policy.get_siglip_features(img_dict).squeeze(0).cpu().numpy()
                state = batch["observation.state"].squeeze(0).cpu().numpy()

                # Step with k=1
                actions, _ = policy.infer(img_dict, batch["observation.state"], task_lang, k=1)
                action = actions[0].cpu().numpy()
                obs, reward, done, info = env.step(action)

                ep_features.append(features)
                ep_states.append(state)
                ep_actions.append(action)
                step += 1

            ep_name = f"task{tid:02d}_ep{ep_idx:04d}"
            np.savez_compressed(
                out_dir / ep_name,
                features=np.stack(ep_features),
                states=np.stack(ep_states),
                actions=np.stack(ep_actions),
                success=np.array([info.get("success", False)]),
            )
            success_str = "✓" if info.get("success") else "✗"
            print(f"  {success_str} {ep_name}: {step} steps")
            ep_count += 1

        env.close()

    print(f"\nCollected {ep_count} episodes → {out_dir}")


if __name__ == "__main__":
    main()
