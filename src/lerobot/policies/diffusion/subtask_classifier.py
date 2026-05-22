"""Frozen DINOv2-S subtask classifier used to condition the Diffusion Policy.

Mirrors the architecture trained by `scripts/train_subtask_classifier.py`:
frozen DINOv2-S + state MLP + prev-subtask embedding -> N-way logits.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SubtaskClassifier(nn.Module):
    def __init__(self, n_classes: int = 8, state_dim: int = 6):
        super().__init__()
        self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self._dino.parameters():
            p.requires_grad_(False)
        feat_dim = 384

        self.state_enc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.subtask_emb = nn.Embedding(n_classes, 16)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 64 + 16, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes),
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor, prev_subtask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_feat = self._dino(img)
        return self.head(torch.cat([img_feat, self.state_enc(state), self.subtask_emb(prev_subtask)], dim=-1))


def load_subtask_classifier(
    path: str | None = None,
    repo_id: str | None = None,
    filename: str = "dino/best.pt",
) -> tuple[SubtaskClassifier, dict]:
    """Load a frozen subtask classifier from a local path or a HF Hub repo.

    Returns (model, meta) where meta carries the classifier's own state_mean / state_std
    (used to renormalize state when running the classifier from inside the diffusion policy).
    """
    if path is None:
        if repo_id is None:
            raise ValueError("subtask classifier: provide either `path` or `repo_id`")
        from huggingface_hub import hf_hub_download
        log.info(f"Downloading subtask classifier from {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Training script wraps the backbone as `backbone._m.*`; remap to our `_dino.*`.
    sd = {k.replace("backbone._m.", "_dino."): v for k, v in ckpt["model"].items()}
    model = SubtaskClassifier()
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"Loaded subtask classifier (val_acc={ckpt.get('val_acc')})")
    return model, {"state_mean": ckpt.get("state_mean"), "state_std": ckpt.get("state_std")}
