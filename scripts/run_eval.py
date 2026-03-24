"""
LIBERO evaluation: fixed-k baselines vs adaptive-k.

Usage:
  # Fixed k baseline
  python scripts/run_eval.py --mode fixed --k 4 --tasks libero_spatial --n_episodes 50

  # Adaptive k
  python scripts/run_eval.py --mode adaptive --selector checkpoints/selector.pt \
    --tasks libero_spatial --n_episodes 50
"""

import sys, os, argparse, json
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def make_env(task_suite_name: str, task_id: int, img_size: int = 224):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task_bddl = suite.get_task_bddl_file_path(task_id)
    task_lang = suite.get_task(task_id).language

    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_bddl,
        "camera_heights": img_size,
        "camera_widths": img_size,
        "camera_names": ["agentview", "robot0_eye_in_hand", "robot0_robotview"],
    })
    return env, task_lang


def obs_to_batch(obs, task_lang: str, device: str):
    """Convert LIBERO obs dict to π₀-FAST batch format.

    LIBERO obs keys (verified):
      - agentview_image (H, W, 3) uint8
      - robot0_eye_in_hand_image (H, W, 3) uint8
      - robot0_robotview_image (H, W, 3) uint8
      - robot0_joint_pos (7,)
      - robot0_joint_pos_cos (7,)
      - robot0_joint_pos_sin (7,)
      - robot0_joint_vel (7,)
      - robot0_eef_pos (3,)
      - robot0_eef_quat (4,)
      - robot0_gripper_qpos (2,)
      - robot0_gripper_qvel (2,)
      + object state keys (varies by task)

    pi0fast-libero expects observation.state of shape (32,).
    We use: joint_pos(7) + joint_vel(7) + eef_pos(3) + eef_quat(4) + gripper_qpos(2)
            = 23 dims → pad to 32
    """
    import torch

    def img_tensor(key):
        img = obs.get(key)
        if img is None:
            return torch.zeros(1, 3, 224, 224, device=device)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC -> CHW
            img = img.permute(2, 0, 1)
        # Resize to 224x224 if needed
        if img.shape[-2:] != (224, 224):
            import torch.nn.functional as F
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
        return img.unsqueeze(0).to(device)

    # pi0fast-libero image keys (from config.input_features)
    batch = {
        "observation.images.image": img_tensor("agentview_image"),
        "observation.images.image2": img_tensor("robot0_eye_in_hand_image"),
        "observation.images.empty_camera_0": img_tensor("robot0_robotview_image"),
    }

    # Robot state (fixed ordering, pad to 32)
    state_parts = []
    for key in [
        "robot0_joint_pos",      # 7
        "robot0_joint_vel",      # 7
        "robot0_eef_pos",        # 3
        "robot0_eef_quat",       # 4
        "robot0_gripper_qpos",   # 2
    ]:
        val = obs.get(key)
        if val is not None:
            state_parts.append(np.asarray(val, dtype=np.float32).ravel())
    state = np.concatenate(state_parts) if state_parts else np.zeros(32, dtype=np.float32)
    state = state[:32]
    if len(state) < 32:
        state = np.pad(state, (0, 32 - len(state)))

    batch["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)
    return batch


def run_episode(env, policy_wrapper, task_lang, k, device, max_steps=600):
    obs = env.reset()
    done = False
    step = 0
    total_latency = 0.0
    n_calls = 0
    k_seq = []

    while not done and step < max_steps:
        batch = obs_to_batch(obs, task_lang, device)
        img_dict = {key: val for key, val in batch.items() if "observation.images" in key}
        actions, latency = policy_wrapper.infer(
            images=img_dict,
            state=batch["observation.state"],
            task=task_lang,
            k=k,
        )
        total_latency += latency
        n_calls += 1
        k_seq.append(k)

        for i in range(k):
            if done or step >= max_steps:
                break
            action = actions[i].cpu().numpy()
            obs, reward, done, info = env.step(action)
            step += 1

    return {
        "success": bool(info.get("success", False)),
        "steps": step,
        "n_vla_calls": n_calls,
        "mean_latency_ms": total_latency / max(n_calls, 1),
        "total_latency_ms": total_latency,
        "k_sequence": k_seq,
        "mean_k": float(np.mean(k_seq)) if k_seq else k,
    }


def run_adaptive_episode(env, policy_wrapper, k_selector, task_lang, device, max_steps=600):
    from src.selector.model import K_CANDIDATES
    obs = env.reset()
    done = False
    step = 0
    total_latency = 0.0
    n_calls = 0
    k_seq = []

    while not done and step < max_steps:
        batch = obs_to_batch(obs, task_lang, device)

        # Get features for k prediction
        img_dict = {key: val for key, val in batch.items() if "observation.images" in key}
        features = policy_wrapper.get_siglip_features(img_dict)
        state = batch["observation.state"]
        k = k_selector.predict_k(features, state)[0]
        k_seq.append(k)

        actions, latency = policy_wrapper.infer(
            images=img_dict,
            state=state,
            task=task_lang,
            k=k,
        )
        total_latency += latency
        n_calls += 1

        for i in range(k):
            if done or step >= max_steps:
                break
            action = actions[i].cpu().numpy()
            obs, reward, done, info = env.step(action)
            step += 1

    return {
        "success": bool(info.get("success", False)),
        "steps": step,
        "n_vla_calls": n_calls,
        "mean_latency_ms": total_latency / max(n_calls, 1),
        "total_latency_ms": total_latency,
        "k_sequence": k_seq,
        "mean_k": float(np.mean(k_seq)) if k_seq else 0,
    }


def summarize(results: list[dict], label: str) -> dict:
    sr = sum(r["success"] for r in results) / len(results)
    lats = [r["mean_latency_ms"] for r in results]
    mean_k = np.mean([r["mean_k"] for r in results])
    summary = {
        "label": label,
        "n_episodes": len(results),
        "success_rate": round(sr, 4),
        "mean_latency_ms": round(float(np.mean(lats)), 1),
        "p90_latency_ms": round(float(np.percentile(lats, 90)), 1),
        "mean_k": round(float(mean_k), 2),
    }
    print(f"[{label:20s}] SR={sr:.2%}  lat={summary['mean_latency_ms']:.0f}ms  mean_k={mean_k:.1f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "adaptive", "all_fixed"], default="fixed")
    parser.add_argument("--k", type=int, default=4, help="chunk size for fixed mode")
    parser.add_argument("--tasks", default="libero_spatial")
    parser.add_argument("--task_ids", nargs="*", type=int, default=None)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--selector", default=None, help="path to selector checkpoint")
    parser.add_argument("--model_id", default="lerobot/pi0fast-libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="experiments/results/eval_results.json")
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    # Load policy
    from src.policy.pi0fast_wrapper import Pi0FastWrapper
    policy = Pi0FastWrapper(device=args.device, model_id=args.model_id).load()

    # Load tasks
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.tasks]()
    n_tasks = suite.get_num_tasks()
    task_ids = args.task_ids or list(range(n_tasks))
    print(f"Tasks: {args.tasks}, n={len(task_ids)}")

    all_results = []

    if args.mode == "fixed" or args.mode == "all_fixed":
        k_list = [1, 2, 4, 8, 16] if args.mode == "all_fixed" else [args.k]
        for k in k_list:
            ep_results = []
            for tid in task_ids:
                env, task_lang = make_env(args.tasks, tid)
                for _ in range(max(1, args.n_episodes // len(task_ids))):
                    r = run_episode(env, policy, task_lang, k, args.device, args.max_steps)
                    ep_results.append(r)
                env.close()
            all_results.append(summarize(ep_results, f"fixed_k={k}"))

    elif args.mode == "adaptive":
        assert args.selector, "Need --selector for adaptive mode"
        from src.selector.model import KSelectorMLP
        ckpt = torch.load(args.selector, map_location=args.device)
        selector = KSelectorMLP(**{k: v for k, v in ckpt["config"].items()
                                   if k in ["feature_dim", "state_dim", "hidden_dims"]})
        selector.load_state_dict(ckpt["model"])
        selector = selector.to(args.device).eval()

        ep_results = []
        for tid in task_ids:
            env, task_lang = make_env(args.tasks, tid)
            for _ in range(max(1, args.n_episodes // len(task_ids))):
                r = run_adaptive_episode(env, policy, selector, task_lang, args.device, args.max_steps)
                ep_results.append(r)
            env.close()
        all_results.append(summarize(ep_results, "adaptive_k"))

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
