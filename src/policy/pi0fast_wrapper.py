"""
π₀-FAST Policy Wrapper with variable k support.

Wraps lerobot's PI0FastPolicy to support dynamic action chunk size,
and exposes SigLIP image features for the k-Selector.
"""

import sys
import time
import torch
import numpy as np
from typing import Optional

# Remove BitVLA path contamination if present
sys.path = [p for p in sys.path if "BitVLA" not in p and "edge-vla-adaptive" not in p]


class Pi0FastWrapper:
    """
    Wraps PI0FastPolicy with:
    - Variable k (n_action_steps) at inference time
    - SigLIP feature extraction for k-Selector
    - Latency measurement
    """

    MODEL_ID = "lerobot/pi0fast-base"

    def __init__(self, device: str = "cuda:0", model_id: Optional[str] = None):
        self.device = device
        self.model_id = model_id or self.MODEL_ID
        self._policy = None
        self._tokenizer = None

    def load(self):
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        self._policy = PI0FastPolicy.from_pretrained(self.model_id).to(self.device).eval()
        self._tokenizer = self._policy._paligemma_tokenizer
        print(f"[Pi0FastWrapper] loaded {self.model_id} → {self.device}")
        print(f"  memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
        return self

    def _build_batch(self, images: dict[str, torch.Tensor], state: torch.Tensor, task: str) -> dict:
        """Build a batch dict for PI0FastPolicy.select_action."""
        batch = {}
        # Images
        for key, img in images.items():
            batch[key] = img.to(self.device)
        # State
        batch["observation.state"] = state.to(self.device)
        # Language tokens
        tok = self._tokenizer(
            task, return_tensors="pt", padding="max_length",
            truncation=True, max_length=48
        )
        batch["observation.language.tokens"] = tok["input_ids"].to(self.device)
        batch["observation.language.attention_mask"] = tok["attention_mask"].bool().to(self.device)
        batch["task"] = [task]
        return batch

    def infer(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        task: str,
        k: int,
    ) -> tuple[torch.Tensor, float]:
        """
        Run inference with chunk size k.

        Returns:
            actions: [k, action_dim] tensor
            latency_ms: float
        """
        assert self._policy is not None, "Call .load() first"
        batch = self._build_batch(images, state, task)

        # Use predict_action_chunk to get k actions in one forward pass
        # Returns [B, chunk_size, action_dim]; we take first k steps
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            action_chunk = self._policy.predict_action_chunk(batch)  # [B, chunk_size, 7]
        torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000

        # action_chunk: [B, chunk_size, 7] → take first k steps → [k, 7]
        actions = action_chunk[0, :k, :]  # [k, action_dim]
        return actions, latency_ms

    def get_siglip_features(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract pooled SigLIP features (for k-Selector input).
        Returns: [B, feature_dim]
        """
        assert self._policy is not None
        with torch.inference_mode():
            img_tensors = [v.to(self.device) for v in images.values()]
            img_stack = torch.stack(img_tensors, dim=1)  # [B, n_cams, C, H, W]
            # Access the SigLIP vision tower
            vision_tower = self._policy.model.paligemma_with_expert.paligemma.model.vision_tower
            B, N, C, H, W = img_stack.shape
            flat = img_stack.view(B * N, C, H, W)
            feats = vision_tower(flat).last_hidden_state  # [B*N, seq, dim]
            feats = feats.mean(dim=1)  # pool: [B*N, dim]
            feats = feats.view(B, N, -1).mean(dim=1)  # avg over cams: [B, dim]
        return feats
