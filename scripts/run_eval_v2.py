"""
LIBERO evaluation v2: uses lerobot's native LiberoEnv (proper obs format).

Usage:
  # Fixed k baseline
  PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 \
  python -u scripts/run_eval_v2.py \
    --mode all_fixed \
    --model_id lerobot/pi0fast-libero \
    --suite libero_spatial \
    --n_episodes 50

  # Single k test
  python -u scripts/run_eval_v2.py --mode fixed --k 4 --n_episodes 10
"""

import sys, os, argparse, json, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_policy(model_id: str, device: str):
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    policy = PI0FastPolicy.from_pretrained(model_id).to(device).eval()
    print(f"[policy] loaded {model_id} → {device}  ({torch.cuda.memory_allocated(device)/1e9:.2f} GB)")
    return policy


def infer_chunk(policy, batch: dict, k: int, device: str) -> tuple[np.ndarray, float]:
    """
    Run one predict_action_chunk and return (actions [k, 7], latency_ms).
    Temporarily sets chunk_size=k so FAST generates exactly k steps.
    """
    orig_chunk = policy.config.chunk_size
    policy.config.chunk_size = k

    # Move batch to device
    dev_batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val
                 for key, val in batch.items()}

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(dev_batch)  # [B, k, 7]
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000

    policy.config.chunk_size = orig_chunk
    actions = action_chunk[0, :k, :].cpu().numpy()  # [k, 7]
    return actions, latency_ms


def obs_to_policy_batch(obs: dict, tokenizer, task_lang: str, device: str) -> dict:
    """
    Convert LiberoEnv obs (pixels_agent_pos mode) to policy batch.

    LiberoEnv obs structure (verified):
      obs["pixels"]["image"]           (224, 224, 3) uint8  ← agentview
      obs["pixels"]["image2"]          (224, 224, 3) uint8  ← wrist
      obs["robot_state"]["joints"]["pos"]  (7,)
      obs["robot_state"]["joints"]["vel"]  (7,)
      obs["robot_state"]["eef"]["pos"]     (3,)
      obs["robot_state"]["eef"]["quat"]    (4,)
      obs["robot_state"]["gripper"]["qpos"] (2,)
      obs["robot_state"]["gripper"]["qvel"] (2,)
    """
    import torch.nn.functional as F

    batch = {}

    def img_tensor(img_np):
        t = torch.from_numpy(img_np.copy()).float() / 255.0  # HWC float
        t = t.permute(2, 0, 1)  # CHW
        if t.shape[-2:] != (224, 224):
            t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear",
                              align_corners=False).squeeze(0)
        return t.unsqueeze(0)   # [1, 3, 224, 224]

    pixels = obs.get("pixels", {})
    batch["observation.images.image"] = img_tensor(
        pixels.get("image", np.zeros((224, 224, 3), dtype=np.uint8))
    )
    batch["observation.images.image2"] = img_tensor(
        pixels.get("image2", np.zeros((224, 224, 3), dtype=np.uint8))
    )
    # pi0fast-libero expects empty_camera_0 (send zeros)
    batch["observation.images.empty_camera_0"] = torch.zeros(1, 3, 224, 224)

    # Robot state: joints.pos(7) + joints.vel(7) + eef.pos(3) + eef.quat(4) + gripper.qpos(2) = 23 → pad to 32
    rs = obs.get("robot_state", {})
    state_parts = []
    for path in [("joints", "pos"), ("joints", "vel"), ("eef", "pos"),
                 ("eef", "quat"), ("gripper", "qpos")]:
        v = rs
        for key in path:
            v = v.get(key, None) if isinstance(v, dict) else None
        if v is not None:
            state_parts.append(np.asarray(v, dtype=np.float32).ravel())
    state = np.concatenate(state_parts) if state_parts else np.zeros(23, dtype=np.float32)
    state = state[:32]
    if len(state) < 32:
        state = np.pad(state, (0, 32 - len(state)))
    batch["observation.state"] = torch.from_numpy(state).float().unsqueeze(0)  # [1, 32]

    # Language tokens
    tok = tokenizer(task_lang, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=48)
    batch["observation.language.tokens"] = tok["input_ids"]
    batch["observation.language.attention_mask"] = tok["attention_mask"].bool()
    batch["task"] = [task_lang]

    return batch


def run_episode(env, policy, tokenizer, task_lang: str, k: int, device: str,
                max_steps: int = 300) -> dict:
    obs, _ = env.reset()
    done = False
    step = 0
    total_lat = 0.0
    n_calls = 0

    while not done and step < max_steps:
        batch = obs_to_policy_batch(obs, tokenizer, task_lang, device)
        actions, lat_ms = infer_chunk(policy, batch, k, device)
        total_lat += lat_ms
        n_calls += 1

        for i in range(k):
            if done or step >= max_steps:
                break
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            step += 1

    success = bool(info.get("is_success", info.get("success", False)))
    return {
        "success": success,
        "steps": step,
        "n_vla_calls": n_calls,
        "mean_latency_ms": total_lat / max(n_calls, 1),
    }


def summarize(results: list[dict], label: str) -> dict:
    sr = sum(r["success"] for r in results) / len(results)
    lats = [r["mean_latency_ms"] for r in results]
    d = {
        "label": label,
        "n_episodes": len(results),
        "success_rate": round(sr, 4),
        "mean_latency_ms": round(float(np.mean(lats)), 1),
        "p90_latency_ms": round(float(np.percentile(lats, 90)), 1),
    }
    print(f"[{label:20s}] SR={sr:.1%}  lat={d['mean_latency_ms']:.0f}ms  n={len(results)}")
    return d


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--mode", choices=["fixed", "all_fixed"], default="all_fixed")
    parser_arg.add_argument("--k", type=int, default=4)
    parser_arg.add_argument("--suite", default="libero_spatial",
                            choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser_arg.add_argument("--task_ids", nargs="*", type=int, default=None)
    parser_arg.add_argument("--n_episodes", type=int, default=50,
                            help="Total episodes across all tasks")
    parser_arg.add_argument("--model_id", default="lerobot/pi0fast-libero")
    parser_arg.add_argument("--device", default="cuda:0")
    parser_arg.add_argument("--max_steps", type=int, default=300)
    parser_arg.add_argument("--output", default="experiments/results/fixed_k_spatial.json")
    args = parser_arg.parse_args()

    # Load policy & tokenizer
    policy = load_policy(args.model_id, args.device)
    tokenizer = policy._paligemma_tokenizer

    # Setup libero suite
    from libero.libero import benchmark
    suite_cls = benchmark.get_benchmark_dict()[args.suite]
    suite = suite_cls()
    n_tasks = suite.get_num_tasks()
    task_ids = args.task_ids or list(range(n_tasks))
    eps_per_task = max(1, args.n_episodes // len(task_ids))
    print(f"Suite: {args.suite}, tasks={len(task_ids)}, eps_per_task={eps_per_task}")

    # lerobot's LiberoEnv (handles obs formatting automatically)
    from lerobot.envs.libero import LiberoEnv

    k_list = [1, 2, 4, 8, 16] if args.mode == "all_fixed" else [args.k]
    all_results = []

    for k in k_list:
        print(f"\n── k={k} ──────────────────────────────")
        ep_results = []

        for tid in task_ids:
            task_lang = suite.get_task(tid).language
            bddl_file = suite.get_task_bddl_file_path(tid)

            env = LiberoEnv(
                task_suite=suite,
                task_id=tid,
                task_suite_name=args.suite,
                obs_type="pixels_agent_pos",
                camera_name="agentview_image,robot0_eye_in_hand_image",
                observation_width=224,
                observation_height=224,
                n_envs=1,
                episode_length=args.max_steps,
            )

            for ep_idx in range(eps_per_task):
                r = run_episode(env, policy, tokenizer, task_lang, k, args.device, args.max_steps)
                ep_results.append(r)
                status = "✓" if r["success"] else "✗"
                print(f"  {status} task{tid:02d} ep{ep_idx:02d}: {r['steps']}steps  {r['mean_latency_ms']:.0f}ms/chunk")

            env.close()

        all_results.append(summarize(ep_results, f"fixed_k={k}"))

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
