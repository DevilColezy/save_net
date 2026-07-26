"""Hierarchical ViT + dual-LSTM policy for guide correction and UAV control.

The model implements two causal, temporally stateful branches over one shared
single-channel depth-image Vision Transformer:

1. Trend branch (coarse navigation)
   Inputs per frame:
       depth image                 [B, T, 1, H, W]
       raw global guide            [B, T, 4]
           - unit goal direction in FLU: x, y, z
           - clipped normalized goal distance in [0, 1]
       gravity direction in FLU    [B, T, 3]
   Visual input:
       deep/final ViT features with global context
   Outputs per frame:
       horizontal guide logits     [B, T, 13]
           class 0:  RECOVER_LEFT
           classes 1..11: normal horizontal guide bins
           class 12: RECOVER_RIGHT
       vertical guide logits       [B, T, 7]
       normalized guide value      [B, T, 1]
           forced to exactly 0 for either recovery class

2. Control branch (fine obstacle avoidance)
   Inputs per frame:
       discrete corrected guide from Trend
       depth image
       gravity direction in FLU    [B, T, 3]
       current velocity in FLU     [B, T, 3]
       current yaw rate            [B, T, 1]
   Visual input:
       shallow + middle ViT patch features, attended using the corrected guide
       and current flight state as a query
   Outputs per frame:
       command                     [B, T, 4]
           vx_flu, vy_flu, vz_flu, yaw_rate

Both branches have independent LSTM states. Hidden states must be reset at the
start of every episode and whenever frame continuity is broken.

This file intentionally depends only on PyTorch. It does not require timm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LSTMState = Tuple[Tensor, Tensor]


@dataclass(frozen=True)
class TrendControlConfig:
    """Configuration for :class:`HierarchicalTrendControlPolicy`.

    The default dimensions target a compact real-time model. ``image_size`` is
    the depth resolution expected during training. At inference, other sizes
    divisible by ``patch_size`` are accepted; positional embeddings are
    interpolated automatically.
    """

    image_size: Tuple[int, int] = (120, 160)
    in_channels: int = 1
    patch_size: int = 8

    vit_embed_dim: int = 128
    vit_depth: int = 6
    vit_num_heads: int = 4
    vit_mlp_ratio: float = 2.0
    vit_dropout: float = 0.05

    # Zero-based Transformer block indices used by Control. Early/middle blocks
    # retain more local obstacle geometry than the final global representation.
    control_feature_layers: Tuple[int, ...] = (1, 3)

    trend_visual_dim: int = 160
    control_visual_dim: int = 128
    auxiliary_feature_dim: int = 64
    guide_feature_dim: int = 72

    trend_lstm_input_dim: int = 256
    trend_lstm_hidden_dim: int = 256
    trend_lstm_layers: int = 2

    control_lstm_input_dim: int = 256
    control_lstm_hidden_dim: int = 256
    control_lstm_layers: int = 2

    horizontal_classes: int = 13
    vertical_classes: int = 7
    recover_left_index: int = 0
    recover_right_index: int = 12

    horizontal_embedding_dim: int = 32
    vertical_embedding_dim: int = 24
    guide_value_embedding_dim: int = 16

    # Signed state inputs are normalized to [-1, 1] inside the model. Scales
    # include 20% headroom over the nominal 2.5 m/s and 2.0 rad/s limits.
    velocity_input_scale: Tuple[float, float, float] = (3.0, 3.0, 3.0)
    yaw_rate_input_scale: float = 2.4

    # FLU command limits. Translation is first tanh-scaled per axis and then
    # projected into the unit L2 ball, matching the expert's vector-speed
    # constraint instead of allowing sqrt(3) times the configured speed.
    max_vx_flu: float = 2.5
    max_vy_flu: float = 2.5
    max_vz_flu: float = 2.5
    max_yaw_rate: float = 2.0
    max_recovery_yaw_rate: float = 0.8
    enforce_translation_norm: bool = True

    # Set +1 when positive yaw rate means left turn in the deployment interface;
    # set -1 if the project's convention is reversed.
    recover_left_yaw_sign: float = 1.0

    enforce_recovery_control: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` for inconsistent architecture settings."""
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive integers")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image_size dimensions must be divisible by patch_size")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.vit_embed_dim <= 0 or self.control_visual_dim <= 0:
            raise ValueError("visual embedding dimensions must be positive")
        if self.vit_num_heads <= 0:
            raise ValueError("vit_num_heads must be positive")
        if self.vit_embed_dim % self.vit_num_heads != 0:
            raise ValueError("vit_embed_dim must be divisible by vit_num_heads")
        if self.control_visual_dim % self.vit_num_heads != 0:
            raise ValueError("control_visual_dim must be divisible by vit_num_heads")
        if self.vit_depth <= 0:
            raise ValueError("vit_depth must be positive")
        if self.vit_mlp_ratio <= 0.0:
            raise ValueError("vit_mlp_ratio must be positive")
        if not 0.0 <= self.vit_dropout < 1.0:
            raise ValueError("vit_dropout must be in [0, 1)")
        if not self.control_feature_layers:
            raise ValueError("control_feature_layers must not be empty")
        feature_layers = tuple(self.control_feature_layers)
        if tuple(sorted(set(feature_layers))) != feature_layers:
            raise ValueError(
                "control_feature_layers must contain unique indices in increasing order"
            )
        if min(self.control_feature_layers) < 0:
            raise ValueError("control_feature_layers must be non-negative")
        if max(self.control_feature_layers) >= self.vit_depth:
            raise ValueError("control_feature_layers contains an invalid block index")
        positive_dimensions = {
            "trend_visual_dim": self.trend_visual_dim,
            "control_visual_dim": self.control_visual_dim,
            "auxiliary_feature_dim": self.auxiliary_feature_dim,
            "guide_feature_dim": self.guide_feature_dim,
            "trend_lstm_input_dim": self.trend_lstm_input_dim,
            "trend_lstm_hidden_dim": self.trend_lstm_hidden_dim,
            "control_lstm_input_dim": self.control_lstm_input_dim,
            "control_lstm_hidden_dim": self.control_lstm_hidden_dim,
            "horizontal_embedding_dim": self.horizontal_embedding_dim,
            "vertical_embedding_dim": self.vertical_embedding_dim,
            "guide_value_embedding_dim": self.guide_value_embedding_dim,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.trend_lstm_layers <= 0 or self.control_lstm_layers <= 0:
            raise ValueError("LSTM layer counts must be positive")
        if self.horizontal_classes != 13:
            raise ValueError("this policy expects exactly 13 horizontal classes")
        if self.vertical_classes != 7:
            raise ValueError("this policy expects exactly 7 vertical classes")
        if self.recover_left_index == self.recover_right_index:
            raise ValueError("recovery indices must be distinct")
        if not 0 <= self.recover_left_index < self.horizontal_classes:
            raise ValueError("recover_left_index is outside the horizontal classes")
        if not 0 <= self.recover_right_index < self.horizontal_classes:
            raise ValueError("recover_right_index is outside the horizontal classes")
        if self.guide_feature_dim != (
            self.horizontal_embedding_dim
            + self.vertical_embedding_dim
            + self.guide_value_embedding_dim
        ):
            raise ValueError(
                "guide_feature_dim must equal horizontal + vertical + value embedding dims"
            )
        if len(self.velocity_input_scale) != 3:
            raise ValueError("velocity_input_scale must contain exactly 3 values")
        for index, value in enumerate(self.velocity_input_scale):
            if value <= 0.0:
                raise ValueError(
                    f"velocity_input_scale[{index}] must be positive"
                )
        if self.yaw_rate_input_scale <= 0.0:
            raise ValueError("yaw_rate_input_scale must be positive")
        command_limits = {
            "max_vx_flu": self.max_vx_flu,
            "max_vy_flu": self.max_vy_flu,
            "max_vz_flu": self.max_vz_flu,
            "max_yaw_rate": self.max_yaw_rate,
            "max_recovery_yaw_rate": self.max_recovery_yaw_rate,
        }
        for name, value in command_limits.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.max_recovery_yaw_rate > self.max_yaw_rate:
            raise ValueError("max_recovery_yaw_rate must not exceed max_yaw_rate")
        if self.recover_left_yaw_sign not in (-1.0, 1.0):
            raise ValueError("recover_left_yaw_sign must be exactly +1.0 or -1.0")


@dataclass
class PolicyOutput:
    """Model outputs.

    Attributes:
        horizontal_logits:
            ``[B, T, 13]`` unnormalized Trend horizontal scores.
        vertical_logits:
            ``[B, T, 7]`` unnormalized Trend vertical scores. Recovery frames
            remain valid supervision and normally use the center vertical class.
        guide_value_raw:
            ``[B, T, 1]`` sigmoid Trend prediction before recovery gating. Use
            this tensor—not ``guide_value``—for the Trend value regression loss.
        guide_value:
            ``[B, T, 1]`` selected/teacher-forced value consumed by Control. It
            is exactly zero when the selected horizontal class is a recovery class.
        horizontal_one_hot:
            ``[B, T, 13]`` guide representation consumed by Control. During
            training it is straight-through one-hot unless teacher forcing is
            active; during evaluation it is hard argmax one-hot.
        vertical_one_hot:
            ``[B, T, 7]`` normal vertical guide representation. On recovery
            frames it is ignored and a dedicated null vertical embedding is used.
        horizontal_index:
            ``[B, T]`` selected horizontal class index.
        vertical_index:
            ``[B, T]`` selected vertical class index. It remains a valid Trend
            supervision target on recovery frames, although Control ignores it.
        vertical_supervision_mask:
            ``[B, T, 1]`` equals 1 for every model-valid frame, including recovery.
            Combine it with the dataset's frame/loss-valid mask in the trainer.
        vertical_control_valid_mask:
            ``[B, T, 1]`` equals 0 on recovery frames and 1 otherwise. This mask
            describes Control semantics only; it must not mask the Trend loss.
        vertical_valid_mask:
            Backward-compatible alias of ``vertical_supervision_mask``.
        recovery_mask:
            ``[B, T, 1]`` equals 1 for RECOVER_LEFT or RECOVER_RIGHT.
        command_normalized:
            ``[B, T, 4]`` normalized command after optional recovery gating.
        command:
            ``[B, T, 4]`` physical command
            ``[vx_flu, vy_flu, vz_flu, yaw_rate]``.
        trend_state / control_state:
            Final LSTM ``(h, c)`` tuples. Each tensor has shape
            ``[num_layers, B, hidden_dim]``.
    """

    horizontal_logits: Tensor
    vertical_logits: Tensor
    guide_value_raw: Tensor
    guide_value: Tensor

    horizontal_one_hot: Tensor
    vertical_one_hot: Tensor
    horizontal_index: Tensor
    vertical_index: Tensor
    vertical_supervision_mask: Tensor
    vertical_control_valid_mask: Tensor
    vertical_valid_mask: Tensor
    recovery_mask: Tensor

    command_normalized: Tensor
    command: Tensor

    trend_state: LSTMState
    control_state: LSTMState


class MLP(nn.Module):
    """Small LayerNorm-SiLU MLP used for vector-valued inputs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block that exposes its intermediate token output."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MultiLevelDepthViT(nn.Module):
    """Shared ViT backbone returning coarse and fine visual representations.

    The Trend branch receives the final CLS token and final patch mean, which
    have passed through all Transformer blocks and therefore encode global
    obstacle layout and coarse navigation context.

    The Control branch receives patch tokens captured from selected shallow and
    middle blocks. These tokens preserve local depth discontinuities, obstacle
    edges and corridor geometry. They are not pooled here; Control performs a
    guide-conditioned cross-attention pooling later.
    """

    def __init__(self, config: TrendControlConfig) -> None:
        super().__init__()
        self.config = config
        image_h, image_w = config.image_size
        base_grid_h = image_h // config.patch_size
        base_grid_w = image_w // config.patch_size
        if base_grid_h <= 0 or base_grid_w <= 0:
            raise ValueError("patch_size is larger than image_size")

        self.base_grid_size = (base_grid_h, base_grid_w)
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.vit_embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                1 + base_grid_h * base_grid_w,
                config.vit_embed_dim,
            )
        )
        self.input_dropout = nn.Dropout(config.vit_dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.vit_embed_dim,
                    num_heads=config.vit_num_heads,
                    mlp_ratio=config.vit_mlp_ratio,
                    dropout=config.vit_dropout,
                )
                for _ in range(config.vit_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.vit_embed_dim)

        fine_concat_dim = config.vit_embed_dim * len(config.control_feature_layers)
        self.fine_token_projection = nn.Sequential(
            nn.LayerNorm(fine_concat_dim),
            nn.Linear(fine_concat_dim, config.control_visual_dim),
            nn.GELU(),
        )
        self.trend_global_projection = nn.Sequential(
            nn.LayerNorm(config.vit_embed_dim * 2),
            nn.Linear(config.vit_embed_dim * 2, config.trend_visual_dim),
            nn.GELU(),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _interpolated_position_embedding(
        self,
        grid_h: int,
        grid_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Interpolate learned 2-D patch positions for a new image resolution."""
        base_h, base_w = self.base_grid_size
        cls_position = self.position_embedding[:, :1]
        patch_position = self.position_embedding[:, 1:]

        if (grid_h, grid_w) == (base_h, base_w):
            return self.position_embedding.to(device=device, dtype=dtype)

        patch_position = patch_position.reshape(
            1,
            base_h,
            base_w,
            self.config.vit_embed_dim,
        ).permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1,
            grid_h * grid_w,
            self.config.vit_embed_dim,
        )
        return torch.cat((cls_position, patch_position), dim=1).to(
            device=device,
            dtype=dtype,
        )

    def forward(self, depth: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a flattened frame batch.

        Args:
            depth: ``[N, 1, H, W]`` normalized depth frames, where
                ``N = B * T``.

        Returns:
            trend_global_feature:
                ``[N, trend_visual_dim]`` deep/global representation.
            control_fine_tokens:
                ``[N, P, control_visual_dim]`` shallow/middle patch tokens,
                where ``P = (H // patch_size) * (W // patch_size)``.
        """
        if depth.ndim != 4:
            raise ValueError(f"depth must be [N,C,H,W], got {tuple(depth.shape)}")
        if depth.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} depth channels, got {depth.shape[1]}"
            )
        if depth.shape[-2] % self.config.patch_size != 0:
            raise ValueError("depth height must be divisible by patch_size")
        if depth.shape[-1] % self.config.patch_size != 0:
            raise ValueError("depth width must be divisible by patch_size")

        patches = self.patch_embed(depth)
        grid_h, grid_w = patches.shape[-2:]
        patch_tokens = patches.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(depth.shape[0], -1, -1)
        tokens = torch.cat((cls, patch_tokens), dim=1)
        tokens = tokens + self._interpolated_position_embedding(
            grid_h,
            grid_w,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens = self.input_dropout(tokens)

        selected_patch_features = []
        selected_layers = set(self.config.control_feature_layers)
        for layer_index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if layer_index in selected_layers:
                # Exclude CLS: Control needs spatially localized obstacle tokens.
                selected_patch_features.append(tokens[:, 1:, :])

        if len(selected_patch_features) != len(self.config.control_feature_layers):
            raise RuntimeError("failed to capture all requested Control ViT layers")

        final_tokens = self.final_norm(tokens)
        final_cls = final_tokens[:, 0, :]
        final_patch_mean = final_tokens[:, 1:, :].mean(dim=1)
        trend_global_feature = self.trend_global_projection(
            torch.cat((final_cls, final_patch_mean), dim=-1)
        )

        control_fine_tokens = self.fine_token_projection(
            torch.cat(selected_patch_features, dim=-1)
        )
        return trend_global_feature, control_fine_tokens


class GuideConditionedVisualPool(nn.Module):
    """Attend to fine visual patches using Guide + flight state as the query."""

    def __init__(self, visual_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(visual_dim)
        self.token_norm = nn.LayerNorm(visual_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim),
            nn.GELU(),
        )

    def forward(self, query: Tensor, fine_tokens: Tensor) -> Tensor:
        """
        Args:
            query: ``[N, visual_dim]`` Guide/state-conditioned query.
            fine_tokens: ``[N, P, visual_dim]`` local visual tokens.

        Returns:
            ``[N, visual_dim]`` fine-grained obstacle feature.
        """
        query_token = self.query_norm(query).unsqueeze(1)
        normalized_tokens = self.token_norm(fine_tokens)
        pooled, _ = self.attention(
            query_token,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        return self.output(pooled.squeeze(1) + query)


class DiscreteGuideEncoder(nn.Module):
    """Encode discrete Trend output for the Control branch.

    A dedicated eighth vertical embedding represents "vertical direction not
    applicable" during RECOVER_LEFT or RECOVER_RIGHT.
    """

    def __init__(self, config: TrendControlConfig) -> None:
        super().__init__()
        self.config = config
        self.horizontal_embedding = nn.Embedding(
            config.horizontal_classes,
            config.horizontal_embedding_dim,
        )
        self.vertical_embedding = nn.Embedding(
            config.vertical_classes + 1,
            config.vertical_embedding_dim,
        )
        self.recovery_vertical_index = config.vertical_classes
        self.value_encoder = MLP(
            input_dim=1,
            hidden_dim=config.guide_value_embedding_dim,
            output_dim=config.guide_value_embedding_dim,
        )

    def forward(
        self,
        horizontal_one_hot: Tensor,
        vertical_one_hot: Tensor,
        guide_value: Tensor,
        recovery_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            horizontal_one_hot: ``[B,T,13]`` straight-through or hard one-hot.
            vertical_one_hot: ``[B,T,7]`` straight-through or hard one-hot.
            guide_value: ``[B,T,1]`` and exactly zero on recovery frames.
            recovery_mask: ``[B,T,1]``.

        Returns:
            Guide feature ``[B,T,guide_feature_dim]``.
        """
        horizontal_feature = horizontal_one_hot @ self.horizontal_embedding.weight
        normal_vertical_feature = vertical_one_hot @ self.vertical_embedding.weight[
            : self.config.vertical_classes
        ]
        recovery_vertical_feature = self.vertical_embedding.weight[
            self.recovery_vertical_index
        ].view(1, 1, -1)
        vertical_feature = (
            (1.0 - recovery_mask) * normal_vertical_feature
            + recovery_mask * recovery_vertical_feature
        )
        value_feature = self.value_encoder(guide_value)
        return torch.cat(
            (horizontal_feature, vertical_feature, value_feature),
            dim=-1,
        )


class HierarchicalTrendControlPolicy(nn.Module):
    """Shared multi-level ViT with independent Trend and Control LSTMs.

    Training input shapes use ``[B, T, ...]``. For online inference use
    :meth:`forward_step`, which accepts a single frame and returns updated LSTM
    states.

    Depth normalization is deliberately external to the model. A typical input
    is metric depth clipped by the sensor maximum and divided by that maximum,
    giving values in ``[0, 1]``.
    """

    def __init__(self, config: Optional[TrendControlConfig] = None) -> None:
        super().__init__()
        self.config = config or TrendControlConfig()
        self.config.validate()
        cfg = self.config

        self.depth_encoder = MultiLevelDepthViT(cfg)

        # Trend non-image input:
        # raw guide [dir_x, dir_y, dir_z, clipped_distance] + gravity [x,y,z].
        self.trend_aux_encoder = MLP(
            input_dim=7,
            hidden_dim=cfg.auxiliary_feature_dim,
            output_dim=cfg.auxiliary_feature_dim,
        )
        self.trend_fusion = nn.Sequential(
            nn.Linear(
                cfg.trend_visual_dim + cfg.auxiliary_feature_dim,
                cfg.trend_lstm_input_dim,
            ),
            nn.LayerNorm(cfg.trend_lstm_input_dim),
            nn.SiLU(),
        )
        self.trend_lstm = nn.LSTM(
            input_size=cfg.trend_lstm_input_dim,
            hidden_size=cfg.trend_lstm_hidden_dim,
            num_layers=cfg.trend_lstm_layers,
            dropout=0.1 if cfg.trend_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.horizontal_head = nn.Sequential(
            nn.Linear(cfg.trend_lstm_hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.horizontal_classes),
        )
        self.vertical_head = nn.Sequential(
            nn.Linear(cfg.trend_lstm_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, cfg.vertical_classes),
        )
        self.guide_value_head = nn.Sequential(
            nn.Linear(cfg.trend_lstm_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.guide_encoder = DiscreteGuideEncoder(cfg)

        # Control non-image state:
        # gravity [3] + current velocity FLU [3] + current yaw rate [1].
        self.control_state_encoder = MLP(
            input_dim=7,
            hidden_dim=cfg.auxiliary_feature_dim,
            output_dim=cfg.auxiliary_feature_dim,
        )
        self.control_query_encoder = nn.Sequential(
            nn.Linear(
                cfg.guide_feature_dim + cfg.auxiliary_feature_dim,
                cfg.control_visual_dim,
            ),
            nn.LayerNorm(cfg.control_visual_dim),
            nn.SiLU(),
        )
        self.control_visual_pool = GuideConditionedVisualPool(
            visual_dim=cfg.control_visual_dim,
            num_heads=cfg.vit_num_heads,
            dropout=cfg.vit_dropout,
        )
        self.control_fusion = nn.Sequential(
            nn.Linear(
                cfg.control_visual_dim
                + cfg.guide_feature_dim
                + cfg.auxiliary_feature_dim,
                cfg.control_lstm_input_dim,
            ),
            nn.LayerNorm(cfg.control_lstm_input_dim),
            nn.SiLU(),
        )
        self.control_lstm = nn.LSTM(
            input_size=cfg.control_lstm_input_dim,
            hidden_size=cfg.control_lstm_hidden_dim,
            num_layers=cfg.control_lstm_layers,
            dropout=0.1 if cfg.control_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.command_head = nn.Sequential(
            nn.Linear(cfg.control_lstm_hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 4),
        )

        self.register_buffer(
            "command_scale",
            torch.tensor(
                [
                    cfg.max_vx_flu,
                    cfg.max_vy_flu,
                    cfg.max_vz_flu,
                    cfg.max_yaw_rate,
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "velocity_input_scale",
            torch.tensor(cfg.velocity_input_scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "yaw_rate_input_scale",
            torch.tensor([cfg.yaw_rate_input_scale], dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def initial_state(
        num_layers: int,
        batch_size: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LSTMState:
        """Create a zero LSTM state for a new episode."""
        shape = (num_layers, batch_size, hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    @staticmethod
    def detach_state(state: LSTMState) -> LSTMState:
        """Detach a recurrent state for truncated backpropagation through time."""
        return state[0].detach(), state[1].detach()

    @staticmethod
    def _validate_lstm_state(
        name: str,
        state: Optional[LSTMState],
        num_layers: int,
        batch_size: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        """Validate a recurrent state before passing it to ``nn.LSTM``."""
        if state is None:
            return
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError(f"{name} must be a (hidden, cell) tuple")
        hidden, cell = state
        expected_shape = (num_layers, batch_size, hidden_dim)
        if tuple(hidden.shape) != expected_shape or tuple(cell.shape) != expected_shape:
            raise ValueError(
                f"{name} tensors must have shape {expected_shape}, "
                f"got hidden={tuple(hidden.shape)}, cell={tuple(cell.shape)}"
            )
        if hidden.device != device or cell.device != device:
            raise ValueError(f"{name} tensors must be on device {device}")
        if hidden.dtype != cell.dtype:
            raise ValueError(f"{name} hidden and cell tensors must share one dtype")
        if not hidden.is_floating_point() or not cell.is_floating_point():
            raise ValueError(f"{name} tensors must use a floating-point dtype")

    @staticmethod
    def _prepare_teacher_class(
        name: str,
        labels: Tensor,
        expected_shape: Tuple[int, int],
        num_classes: int,
        device: torch.device,
    ) -> Tensor:
        """Move and validate hard teacher class IDs without silent truncation."""
        if tuple(labels.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if labels.is_floating_point():
            rounded = labels.round()
            if not torch.allclose(labels, rounded, atol=0.0, rtol=0.0):
                raise ValueError(f"{name} must contain integer-valued class IDs")
        labels_long = labels.to(device=device, dtype=torch.long)
        if torch.any(labels_long < 0) or torch.any(labels_long >= num_classes):
            raise ValueError(f"{name} contains a class ID outside [0, {num_classes - 1}]")
        return labels_long

    @staticmethod
    def _prepare_teacher_forcing_mask(mask: Tensor, reference: Tensor) -> Tensor:
        """Validate a binary scheduled-sampling mask and move it to the model device."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if tuple(mask.shape) != tuple(reference.shape):
            raise ValueError(
                "teacher_forcing_mask must have shape [B,T] or [B,T,1]"
            )
        mask = mask.to(device=reference.device)
        if mask.is_floating_point() and not torch.isfinite(mask).all():
            raise ValueError("teacher_forcing_mask contains non-finite values")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("teacher_forcing_mask must contain only 0/1 or bool values")
        return mask.to(dtype=reference.dtype)

    @staticmethod
    def _straight_through_one_hot(logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Return argmax indices and straight-through one-hot values."""
        probabilities = logits.softmax(dim=-1)
        indices = probabilities.argmax(dim=-1)
        hard = F.one_hot(indices, num_classes=logits.shape[-1]).to(probabilities.dtype)
        straight_through = hard - probabilities.detach() + probabilities
        return indices, straight_through

    def _select_guide_for_control(
        self,
        horizontal_logits: Tensor,
        vertical_logits: Tensor,
        guide_value_raw: Tensor,
        teacher_horizontal: Optional[Tensor],
        teacher_vertical: Optional[Tensor],
        teacher_guide_value: Optional[Tensor],
        teacher_forcing_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Build the discrete Guide interface consumed by Control.

        Recovery frames still have a valid vertical Trend target (the dataset uses
        the center vertical class), but Control intentionally replaces that class
        with a dedicated recovery/null vertical embedding.

        Teacher tensors are optional:
            teacher_horizontal: ``[B,T]`` integer class IDs.
            teacher_vertical: ``[B,T]`` integer class IDs.
            teacher_guide_value: ``[B,T]`` or ``[B,T,1]`` normalized values.
            teacher_forcing_mask: ``[B,T]`` or ``[B,T,1]`` binary mask.

        If teacher labels are supplied without a mask, teacher forcing is used
        for every frame. A mask without teacher labels is rejected.
        """
        cfg = self.config
        batch_time = tuple(horizontal_logits.shape[:2])
        predicted_h_index, predicted_h_one_hot = self._straight_through_one_hot(
            horizontal_logits
        )
        predicted_v_index, predicted_v_one_hot = self._straight_through_one_hot(
            vertical_logits
        )

        if not self.training:
            # Remove the straight-through gradient term at inference.
            predicted_h_one_hot = F.one_hot(
                predicted_h_index,
                num_classes=cfg.horizontal_classes,
            ).to(device=horizontal_logits.device, dtype=horizontal_logits.dtype)
            predicted_v_one_hot = F.one_hot(
                predicted_v_index,
                num_classes=cfg.vertical_classes,
            ).to(device=vertical_logits.device, dtype=vertical_logits.dtype)

        teacher_flags = (
            teacher_horizontal is not None,
            teacher_vertical is not None,
            teacher_guide_value is not None,
        )
        if any(teacher_flags) and not all(teacher_flags):
            raise ValueError(
                "teacher_horizontal, teacher_vertical and teacher_guide_value "
                "must be provided together"
            )
        use_teacher = all(teacher_flags)
        if not use_teacher and teacher_forcing_mask is not None:
            raise ValueError(
                "teacher_forcing_mask cannot be supplied without teacher labels"
            )

        if use_teacher:
            assert teacher_horizontal is not None
            assert teacher_vertical is not None
            assert teacher_guide_value is not None
            teacher_h_index = self._prepare_teacher_class(
                "teacher_horizontal",
                teacher_horizontal,
                batch_time,
                cfg.horizontal_classes,
                horizontal_logits.device,
            )
            teacher_v_index = self._prepare_teacher_class(
                "teacher_vertical",
                teacher_vertical,
                batch_time,
                cfg.vertical_classes,
                vertical_logits.device,
            )
            teacher_h_one_hot = F.one_hot(
                teacher_h_index,
                num_classes=cfg.horizontal_classes,
            ).to(device=horizontal_logits.device, dtype=horizontal_logits.dtype)
            teacher_v_one_hot = F.one_hot(
                teacher_v_index,
                num_classes=cfg.vertical_classes,
            ).to(device=vertical_logits.device, dtype=vertical_logits.dtype)

            if teacher_guide_value.ndim == 2:
                teacher_guide_value = teacher_guide_value.unsqueeze(-1)
            if tuple(teacher_guide_value.shape) != tuple(guide_value_raw.shape):
                raise ValueError(
                    "teacher_guide_value must have shape [B,T] or [B,T,1]"
                )
            teacher_value = teacher_guide_value.to(
                device=guide_value_raw.device,
                dtype=guide_value_raw.dtype,
            )
            if not torch.isfinite(teacher_value).all():
                raise ValueError("teacher_guide_value contains non-finite values")
            if torch.any(teacher_value < -1e-6) or torch.any(teacher_value > 1.0 + 1e-6):
                raise ValueError("teacher_guide_value must lie in [0, 1]")
            teacher_value = teacher_value.clamp(0.0, 1.0)

            if teacher_forcing_mask is None:
                mask = torch.ones_like(guide_value_raw)
            else:
                mask = self._prepare_teacher_forcing_mask(
                    teacher_forcing_mask, guide_value_raw
                )

            horizontal_one_hot = (
                mask * teacher_h_one_hot + (1.0 - mask) * predicted_h_one_hot
            )
            vertical_one_hot = (
                mask * teacher_v_one_hot + (1.0 - mask) * predicted_v_one_hot
            )
            horizontal_index = horizontal_one_hot.argmax(dim=-1)
            vertical_index = vertical_one_hot.argmax(dim=-1)
            selected_guide_value = (
                mask * teacher_value + (1.0 - mask) * guide_value_raw
            )
        else:
            horizontal_one_hot = predicted_h_one_hot
            vertical_one_hot = predicted_v_one_hot
            horizontal_index = predicted_h_index
            vertical_index = predicted_v_index
            selected_guide_value = guide_value_raw

        left_weight = horizontal_one_hot[
            ..., cfg.recover_left_index : cfg.recover_left_index + 1
        ]
        right_weight = horizontal_one_hot[
            ..., cfg.recover_right_index : cfg.recover_right_index + 1
        ]
        recovery_mask = (left_weight + right_weight).clamp(0.0, 1.0)

        # Recovery has no normal Guide distance, but its vertical Trend label is
        # still supervised. Only the Control branch treats vertical Guide as null.
        guide_value = selected_guide_value.clamp(0.0, 1.0) * (1.0 - recovery_mask)
        vertical_supervision_mask = torch.ones_like(recovery_mask)
        vertical_control_valid_mask = 1.0 - recovery_mask

        return {
            "horizontal_one_hot": horizontal_one_hot,
            "vertical_one_hot": vertical_one_hot,
            "horizontal_index": horizontal_index,
            "vertical_index": vertical_index,
            "guide_value": guide_value,
            "recovery_mask": recovery_mask,
            "vertical_supervision_mask": vertical_supervision_mask,
            "vertical_control_valid_mask": vertical_control_valid_mask,
            # Backward-compatible name. It now means Trend supervision validity.
            "vertical_valid_mask": vertical_supervision_mask,
            "left_weight": left_weight,
            "right_weight": right_weight,
        }

    def _apply_command_constraints(
        self,
        raw_command: Tensor,
        recovery_mask: Tensor,
        left_weight: Tensor,
        right_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Scale command outputs and optionally enforce recovery behavior."""
        cfg = self.config
        normal_command = torch.tanh(raw_command)
        if cfg.enforce_translation_norm:
            translation = normal_command[..., :3]
            translation_norm = translation.norm(p=2, dim=-1, keepdim=True)
            translation = translation / translation_norm.clamp_min(1.0)
            normal_command = torch.cat(
                (translation, normal_command[..., 3:4]), dim=-1
            )

        if not cfg.enforce_recovery_control:
            command_normalized = normal_command
            command = command_normalized * self.command_scale.to(
                dtype=raw_command.dtype,
                device=raw_command.device,
            )
            return command_normalized, command

        translation = normal_command[..., :3] * (1.0 - recovery_mask)
        normal_yaw = normal_command[..., 3:4]

        # In recovery the class determines the yaw sign, while Control predicts
        # only the non-negative magnitude. This prevents left/right sign flips.
        recovery_yaw_magnitude = torch.sigmoid(raw_command[..., 3:4])
        yaw_sign = cfg.recover_left_yaw_sign * (left_weight - right_weight)
        recovery_yaw_normalized = (
            yaw_sign
            * recovery_yaw_magnitude
            * (cfg.max_recovery_yaw_rate / cfg.max_yaw_rate)
        )
        final_yaw_normalized = (
            (1.0 - recovery_mask) * normal_yaw + recovery_yaw_normalized
        )
        command_normalized = torch.cat((translation, final_yaw_normalized), dim=-1)
        command = command_normalized * self.command_scale.to(
            dtype=raw_command.dtype,
            device=raw_command.device,
        )
        return command_normalized, command

    def forward(
        self,
        depth: Tensor,
        raw_guide: Tensor,
        gravity_flu: Tensor,
        velocity_flu: Tensor,
        yaw_rate: Tensor,
        trend_state: Optional[LSTMState] = None,
        control_state: Optional[LSTMState] = None,
        teacher_horizontal: Optional[Tensor] = None,
        teacher_vertical: Optional[Tensor] = None,
        teacher_guide_value: Optional[Tensor] = None,
        teacher_forcing_mask: Optional[Tensor] = None,
    ) -> PolicyOutput:
        """Run a causal sequence through Trend and Control.

        Args:
            depth:
                ``[B,T,1,H,W]`` normalized depth sequence.
            raw_guide:
                ``[B,T,4]`` raw global guide:
                ``[goal_dir_x_flu, goal_dir_y_flu, goal_dir_z_flu,
                clipped_goal_distance_norm]``.
            gravity_flu:
                ``[B,T,3]`` unit gravity direction in FLU.
            velocity_flu:
                ``[B,T,3]`` current translational velocity in FLU.
            yaw_rate:
                ``[B,T,1]`` current angular velocity around FLU z.
            trend_state:
                Optional Trend LSTM ``(h,c)`` from a preceding sequence chunk.
            control_state:
                Optional Control LSTM ``(h,c)`` from a preceding sequence chunk.
            teacher_horizontal / teacher_vertical / teacher_guide_value:
                Optional ground-truth corrected Guide used for Control teacher
                forcing. Class tensors are ``[B,T]`` and guide value is
                ``[B,T]`` or ``[B,T,1]``. Recovery values are gated to zero
                before entering Control. Trend losses must still use logits and
                ``guide_value_raw`` rather than the selected teacher-forced values.
            teacher_forcing_mask:
                Optional ``[B,T]`` or ``[B,T,1]`` mask enabling teacher Guide
                input on selected frames. Omit it to teacher-force all frames
                whenever teacher labels are supplied.

        Returns:
            :class:`PolicyOutput` containing Trend predictions, the discrete
            Guide interface, Control commands and final recurrent states.
        """
        self._validate_sequence_inputs(
            depth,
            raw_guide,
            gravity_flu,
            velocity_flu,
            yaw_rate,
        )
        batch_size, sequence_length = depth.shape[:2]
        self._validate_lstm_state(
            "trend_state",
            trend_state,
            self.config.trend_lstm_layers,
            batch_size,
            self.config.trend_lstm_hidden_dim,
            depth.device,
        )
        self._validate_lstm_state(
            "control_state",
            control_state,
            self.config.control_lstm_layers,
            batch_size,
            self.config.control_lstm_hidden_dim,
            depth.device,
        )

        flat_depth = depth.reshape(
            batch_size * sequence_length,
            depth.shape[2],
            depth.shape[3],
            depth.shape[4],
        )
        trend_visual_flat, control_tokens_flat = self.depth_encoder(flat_depth)
        trend_visual = trend_visual_flat.reshape(batch_size, sequence_length, -1)

        trend_aux = self.trend_aux_encoder(
            torch.cat((raw_guide, gravity_flu), dim=-1)
        )
        trend_input = self.trend_fusion(
            torch.cat((trend_visual, trend_aux), dim=-1)
        )
        trend_sequence, next_trend_state = self.trend_lstm(
            trend_input,
            trend_state,
        )

        horizontal_logits = self.horizontal_head(trend_sequence)
        vertical_logits = self.vertical_head(trend_sequence)
        guide_value_raw = torch.sigmoid(self.guide_value_head(trend_sequence))

        selected_guide = self._select_guide_for_control(
            horizontal_logits=horizontal_logits,
            vertical_logits=vertical_logits,
            guide_value_raw=guide_value_raw,
            teacher_horizontal=teacher_horizontal,
            teacher_vertical=teacher_vertical,
            teacher_guide_value=teacher_guide_value,
            teacher_forcing_mask=teacher_forcing_mask,
        )

        guide_feature = self.guide_encoder(
            horizontal_one_hot=selected_guide["horizontal_one_hot"],
            vertical_one_hot=selected_guide["vertical_one_hot"],
            guide_value=selected_guide["guide_value"],
            recovery_mask=selected_guide["recovery_mask"],
        )
        velocity_normalized = (
            velocity_flu
            / self.velocity_input_scale.to(
                device=velocity_flu.device, dtype=velocity_flu.dtype,
            )
        ).clamp(-1.0, 1.0)
        yaw_rate_normalized = (
            yaw_rate
            / self.yaw_rate_input_scale.to(
                device=yaw_rate.device, dtype=yaw_rate.dtype,
            )
        ).clamp(-1.0, 1.0)
        control_state_feature = self.control_state_encoder(
            torch.cat(
                (gravity_flu, velocity_normalized, yaw_rate_normalized),
                dim=-1,
            )
        )

        guide_state_feature = torch.cat(
            (guide_feature, control_state_feature),
            dim=-1,
        )
        visual_query = self.control_query_encoder(guide_state_feature).reshape(
            batch_size * sequence_length,
            self.config.control_visual_dim,
        )
        fine_obstacle_feature = self.control_visual_pool(
            query=visual_query,
            fine_tokens=control_tokens_flat,
        ).reshape(batch_size, sequence_length, -1)

        control_input = self.control_fusion(
            torch.cat(
                (
                    fine_obstacle_feature,
                    guide_feature,
                    control_state_feature,
                ),
                dim=-1,
            )
        )
        control_sequence, next_control_state = self.control_lstm(
            control_input,
            control_state,
        )
        raw_command = self.command_head(control_sequence)
        command_normalized, command = self._apply_command_constraints(
            raw_command=raw_command,
            recovery_mask=selected_guide["recovery_mask"],
            left_weight=selected_guide["left_weight"],
            right_weight=selected_guide["right_weight"],
        )

        return PolicyOutput(
            horizontal_logits=horizontal_logits,
            vertical_logits=vertical_logits,
            guide_value_raw=guide_value_raw,
            guide_value=selected_guide["guide_value"],
            horizontal_one_hot=selected_guide["horizontal_one_hot"],
            vertical_one_hot=selected_guide["vertical_one_hot"],
            horizontal_index=selected_guide["horizontal_index"],
            vertical_index=selected_guide["vertical_index"],
            vertical_supervision_mask=selected_guide["vertical_supervision_mask"],
            vertical_control_valid_mask=selected_guide[
                "vertical_control_valid_mask"
            ],
            vertical_valid_mask=selected_guide["vertical_valid_mask"],
            recovery_mask=selected_guide["recovery_mask"],
            command_normalized=command_normalized,
            command=command,
            trend_state=next_trend_state,
            control_state=next_control_state,
        )

    def forward_step(
        self,
        depth: Tensor,
        raw_guide: Tensor,
        gravity_flu: Tensor,
        velocity_flu: Tensor,
        yaw_rate: Tensor,
        trend_state: Optional[LSTMState] = None,
        control_state: Optional[LSTMState] = None,
    ) -> PolicyOutput:
        """Online single-frame inference.

        Args:
            depth: ``[B,1,H,W]``.
            raw_guide: ``[B,4]``.
            gravity_flu: ``[B,3]``.
            velocity_flu: ``[B,3]``.
            yaw_rate: ``[B,1]``.
            trend_state / control_state: recurrent states from the previous frame.

        Returns:
            Same fields as :meth:`forward`, each with a sequence dimension of 1.
            Feed the returned states into the next call. Reset both states to
            ``None`` at every episode boundary or frame discontinuity.
        """
        if depth.ndim != 4:
            raise ValueError("forward_step depth must have shape [B,C,H,W]")
        batch_size = depth.shape[0]
        expected = {
            "raw_guide": (batch_size, 4),
            "gravity_flu": (batch_size, 3),
            "velocity_flu": (batch_size, 3),
            "yaw_rate": (batch_size, 1),
        }
        actual = {
            "raw_guide": tuple(raw_guide.shape),
            "gravity_flu": tuple(gravity_flu.shape),
            "velocity_flu": tuple(velocity_flu.shape),
            "yaw_rate": tuple(yaw_rate.shape),
        }
        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise ValueError(
                    f"forward_step {name} must have shape {expected_shape}, "
                    f"got {actual[name]}"
                )
        return self.forward(
            depth=depth.unsqueeze(1),
            raw_guide=raw_guide.unsqueeze(1),
            gravity_flu=gravity_flu.unsqueeze(1),
            velocity_flu=velocity_flu.unsqueeze(1),
            yaw_rate=yaw_rate.unsqueeze(1),
            trend_state=trend_state,
            control_state=control_state,
        )

    @staticmethod
    def _validate_sequence_inputs(
        depth: Tensor,
        raw_guide: Tensor,
        gravity_flu: Tensor,
        velocity_flu: Tensor,
        yaw_rate: Tensor,
    ) -> None:
        if depth.ndim != 5:
            raise ValueError(f"depth must be [B,T,C,H,W], got {tuple(depth.shape)}")
        batch_size, sequence_length = depth.shape[:2]
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("depth batch and sequence dimensions must be non-empty")
        expected = {
            "raw_guide": (batch_size, sequence_length, 4),
            "gravity_flu": (batch_size, sequence_length, 3),
            "velocity_flu": (batch_size, sequence_length, 3),
            "yaw_rate": (batch_size, sequence_length, 1),
        }
        tensors = {
            "depth": depth,
            "raw_guide": raw_guide,
            "gravity_flu": gravity_flu,
            "velocity_flu": velocity_flu,
            "yaw_rate": yaw_rate,
        }
        for name, expected_shape in expected.items():
            actual_shape = tuple(tensors[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {actual_shape}"
                )
        reference_device = depth.device
        for name, tensor in tensors.items():
            if tensor.device != reference_device:
                raise ValueError(
                    f"all model inputs must share device {reference_device}; "
                    f"{name} is on {tensor.device}"
                )
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{name} must use a floating-point dtype; normalize/cast it "
                    "before calling the model"
                )


if __name__ == "__main__":
    # Lightweight shape smoke check; this block is not required by training.
    torch.manual_seed(7)
    config = TrendControlConfig(
        image_size=(120, 160),
        vit_embed_dim=96,
        vit_num_heads=4,
        vit_depth=4,
        control_feature_layers=(0, 2),
        trend_visual_dim=128,
        control_visual_dim=96,
        trend_lstm_hidden_dim=128,
        control_lstm_hidden_dim=128,
        trend_lstm_input_dim=192,
        control_lstm_input_dim=192,
        trend_lstm_layers=1,
        control_lstm_layers=1,
    )
    model = HierarchicalTrendControlPolicy(config)
    model.eval()

    batch, time = 2, 3
    output = model(
        depth=torch.rand(batch, time, 1, 120, 160),
        raw_guide=torch.rand(batch, time, 4),
        gravity_flu=F.normalize(torch.rand(batch, time, 3), dim=-1),
        velocity_flu=torch.randn(batch, time, 3),
        yaw_rate=torch.randn(batch, time, 1),
    )
    print("horizontal_logits:", tuple(output.horizontal_logits.shape))
    print("vertical_logits:  ", tuple(output.vertical_logits.shape))
    print("guide_value:       ", tuple(output.guide_value.shape))
    print("command:           ", tuple(output.command.shape))
