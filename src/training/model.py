"""
IndiaVLAModel: Vision-Language-Action model for egocentric head-pose prediction.

Architecture:
  - Vision encoder:   MobileNetV3-small (pretrained, optionally fine-tuned)
  - Language encoder: TinyLlama / Phi-3-mini embedding layer (optional, offline-safe)
  - IMU encoder:      3-layer MLP
  - Fusion:           Multi-head cross-attention (vision tokens attend to language)
  - Action head:      MLP → 7-DOF continuous action (head pose delta)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

logger = logging.getLogger(__name__)

_ACTION_DIM    = 7
_VISION_DIM    = 576    # MobileNetV3-small last feature dim
_DEFAULT_HIDDEN = 512


# ── Sub-modules ─────────────────────────────────────────────────────────────────

class IMUEncoder(nn.Module):
    """Three-layer MLP that encodes a 10-dim IMU vector to hidden_dim."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = _DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionEncoder(nn.Module):
    """
    MobileNetV3-small backbone stripped of the classifier head.

    Outputs a spatial feature map that is then flattened/averaged for use
    in cross-attention.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        out_dim: int = _DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mv3 = mobilenet_v3_small(weights=weights)

        # Strip classifier head; keep feature extractor
        self.features   = mv3.features
        self.avgpool    = mv3.avgpool
        self.project    = nn.Linear(_VISION_DIM, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)  float32

        Returns
        -------
        (B, out_dim) visual feature vector
        """
        feats = self.features(x)       # (B, 576, 7, 7) for 224×224 input
        feats = self.avgpool(feats)    # (B, 576, 1, 1)
        feats = feats.flatten(1)       # (B, 576)
        feats = self.project(feats)    # (B, out_dim)
        return self.layer_norm(feats)


class LanguageEncoder(nn.Module):
    """
    Language encoder: uses a pretrained LLM's embedding layer.

    Falls back to a trainable embedding table if the model is unavailable offline.

    Parameters
    ----------
    model_name : str   HF model name.
    out_dim : int      output dimension.
    max_vocab : int    vocabulary size for the fallback embedding.
    freeze : bool      freeze LLM weights (LoRA-style training not included here).
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        out_dim: int = _DEFAULT_HIDDEN,
        max_vocab: int = 32000,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim    = out_dim
        self._llm_mode  = False

        try:
            from transformers import AutoModel
            logger.info("Loading language encoder from %s …", model_name)
            lm = AutoModel.from_pretrained(model_name, local_files_only=False)
            self._embed = lm.get_input_embeddings()
            embed_dim   = self._embed.embedding_dim

            if freeze:
                for p in self._embed.parameters():
                    p.requires_grad = False

            self._project   = nn.Linear(embed_dim, out_dim)
            self._layer_norm = nn.LayerNorm(out_dim)
            self._llm_mode   = True
            logger.info("Language encoder loaded (embed_dim=%d).", embed_dim)
        except Exception as exc:
            logger.warning(
                "Language model '%s' not available (%s). "
                "Using fallback trainable embedding.",
                model_name, exc,
            )
            embed_dim        = min(out_dim, 256)
            self._embed      = nn.Embedding(max_vocab, embed_dim, padding_idx=0)
            self._project    = nn.Linear(embed_dim, out_dim)
            self._layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, L) long
        attention_mask : (B, L) long, optional

        Returns
        -------
        (B, out_dim)  mean-pooled language feature vector
        """
        emb = self._embed(input_ids)   # (B, L, embed_dim)
        if attention_mask is not None:
            mask  = attention_mask.unsqueeze(-1).float()
            emb   = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            emb = emb.mean(1)

        out = self._project(emb)
        return self._layer_norm(out)


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention: vision tokens (query) attend to language (key/value).
    """

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_HIDDEN,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn       = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ff         = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ff_norm    = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        vision   : (B, hidden_dim)
        language : (B, hidden_dim)

        Returns
        -------
        (B, hidden_dim) fused feature.
        """
        # Expand to sequence length 1 for MultiheadAttention
        q = vision.unsqueeze(1)    # (B, 1, D)
        k = language.unsqueeze(1)  # (B, 1, D)
        v = k

        attn_out, _ = self.attn(q, k, v)   # (B, 1, D)
        attn_out    = attn_out.squeeze(1)   # (B, D)

        x = self.layer_norm(attn_out + vision)
        x = self.ff_norm(self.ff(x) + x)
        return x


class ActionHead(nn.Module):
    """
    MLP action decoder predicting a 7-DOF head-pose delta:
    [dx, dy, dz, dqx, dqy, dqz, dqw]
    """

    def __init__(
        self,
        input_dim: int = _DEFAULT_HIDDEN,
        action_dim: int = _ACTION_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main model ───────────────────────────────────────────────────────────────────

class IndiaVLAModel(nn.Module):
    """
    End-to-end Vision-Language-Action model.

    Inputs
    ------
    images          : (B, 3, H, W) float32 — normalized RGB
    language_tokens : (B, L) long   — tokenized instruction
    imu             : (B, 10) float32
    attention_mask  : (B, L) long, optional

    Output
    ------
    action_logits   : (B, 7) float32
    """

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_HIDDEN,
        action_dim: int = _ACTION_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        freeze_vision: bool = False,
        language_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_language: bool = True,
        imu_input_dim: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.vision_encoder = VisionEncoder(
            pretrained=pretrained_vision,
            freeze_backbone=freeze_vision,
            out_dim=hidden_dim,
        )
        self.language_encoder = LanguageEncoder(
            model_name=language_model_name,
            out_dim=hidden_dim,
            freeze=freeze_language,
        )
        self.imu_encoder = IMUEncoder(
            input_dim=imu_input_dim,
            hidden_dim=hidden_dim,
        )
        self.fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Combine vision+language with IMU
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_head = ActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        imu: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vis_feat  = self.vision_encoder(images)
        lang_feat = self.language_encoder(language_tokens, attention_mask)
        imu_feat  = self.imu_encoder(imu)

        fused     = self.fusion(vis_feat, lang_feat)                        # (B, D)
        combined  = self.combine(torch.cat([fused, imu_feat], dim=-1))     # (B, D)
        actions   = self.action_head(combined)                              # (B, 7)
        return actions

    # ── Serialization ─────────────────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and config to a directory."""
        import json
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), out / "model.pt")
        config = {
            "hidden_dim":           self.hidden_dim,
            "action_dim":           self.action_dim,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2))
        logger.info("Model saved to %s", out)

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        map_location: Optional[Any] = None,
        **kwargs,
    ) -> "IndiaVLAModel":
        """Load model weights and config from a directory."""
        import json
        p = Path(load_dir)
        config_path = p / "config.json"

        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())

        config.update(kwargs)
        model = cls(**config)

        weights_path = p / "model.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=map_location)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded weights from %s", weights_path)
        else:
            logger.warning("No model.pt found at %s — using random weights.", load_dir)

        return model

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts per sub-module."""
        return {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in [
                ("vision",   self.vision_encoder),
                ("language", self.language_encoder),
                ("imu",      self.imu_encoder),
                ("fusion",   self.fusion),
                ("action",   self.action_head),
            ]
        }
