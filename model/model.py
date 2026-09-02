#!/usr/bin/env python3
"""Causal depth-to-control policy for il_dataset schema v25.

The visual trunk follows ViTFly's useful design choice: two overlapping
Mix-Transformer stages followed by an LSTM.  Unlike the old model in this
package, this policy consumes no privileged planner candidates or modes.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ViTFlyPolicyConfig:
    image_height: int = 60
    image_width: int = 90
    state_dim: int = 7
    stage_dims: Tuple[int, int] = (32, 64)
    stage_depths: Tuple[int, int] = (2, 2)
    stage_heads: Tuple[int, int] = (1, 2)
    sr_ratios: Tuple[int, int] = (8, 4)
    visual_dim: int = 256
    state_hidden_dim: int = 96
    lstm_hidden_dim: int = 192
    lstm_layers: int = 3
    dropout: float = 0.1
    command_scale: Tuple[float, float, float, float] = (2.5, 2.5, 2.5, 1.5)
    # Visual-feature scale alignment (2026-08-26): the Mix-Transformer visual
    # embedding sat ~0.13 std while the state encoder sat ~0.35 std, so the
    # LSTM was driven almost entirely by the state branch and the depth
    # response vanished.  Scale the visual embedding to match the state
    # branch so the policy actually reads the depth image.
    visual_scale: float = 3.0

    def validate(self) -> None:
        if self.state_dim != 7:
            raise ValueError(
                "30 Hz policy requires exactly 7 non-visual inputs "
                "(gravity 3 + goal direction 3 + goal distance 1); "
                "velocity/yaw_rate inputs removed 2026-08-26")
        if self.image_height < 16 or self.image_width < 16:
            raise ValueError("image size is too small")
        if not (len(self.stage_dims) == len(self.stage_depths) ==
                len(self.stage_heads) == len(self.sr_ratios) == 2):
            raise ValueError("exactly two transformer stages are required")
        if len(self.command_scale) != 4 or min(self.command_scale) <= 0:
            raise ValueError("command_scale must contain four positive values")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PolicyOutput:
    command: Tensor
    normalized_command: Tensor
    hidden: Tuple[Tensor, Tensor]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernel: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=kernel, stride=stride,
            padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x)


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio) \
            if sr_ratio > 1 else None
        self.sr_norm = nn.LayerNorm(dim) if sr_ratio > 1 else nn.Identity()
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.q(tokens).reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        source = x
        if self.sr is not None and h >= self.sr.kernel_size[0] and \
                w >= self.sr.kernel_size[1]:
            source = self.sr(source)
        source = self.sr_norm(source.flatten(2).transpose(1, 2))
        kv = self.kv(source).reshape(
            b, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        return self.drop(self.out(out)).transpose(1, 2).reshape(b, c, h, w)


class MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.dw(self.fc1(x))))))


class MixTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = EfficientSelfAttention(dim, heads, sr_ratio, dropout)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = MixFFN(dim, 4, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class VisualEncoder(nn.Module):
    def __init__(self, cfg: ViTFlyPolicyConfig):
        super().__init__()
        d1, d2 = cfg.stage_dims
        self.patch1 = OverlapPatchEmbed(1, d1, 7, 4)
        self.stage1 = nn.Sequential(*[
            MixTransformerBlock(d1, cfg.stage_heads[0], cfg.sr_ratios[0], cfg.dropout)
            for _ in range(cfg.stage_depths[0])])
        self.patch2 = OverlapPatchEmbed(d1, d2, 3, 2)
        self.stage2 = nn.Sequential(*[
            MixTransformerBlock(d2, cfg.stage_heads[1], cfg.sr_ratios[1], cfg.dropout)
            for _ in range(cfg.stage_depths[1])])
        self.proj = nn.Sequential(
            nn.Linear(d1 + d2, cfg.visual_dim), nn.LayerNorm(cfg.visual_dim),
            nn.GELU(), nn.Dropout(cfg.dropout))

    def forward(self, depth: Tensor) -> Tensor:
        x1 = self.stage1(self.patch1(depth))
        x2 = self.stage2(self.patch2(x1))
        pooled = torch.cat((x1.mean((-2, -1)), x2.mean((-2, -1))), dim=-1)
        return self.proj(pooled)


class ViTFlyLSTMPolicy(nn.Module):
    """Single-rate causal policy: depth + v25 state -> [vx, vy, vz, yaw_rate]."""

    def __init__(self, config: Optional[ViTFlyPolicyConfig] = None):
        super().__init__()
        self.config = config or ViTFlyPolicyConfig()
        self.config.validate()
        self.visual = VisualEncoder(self.config)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.state_hidden_dim),
            nn.LayerNorm(self.config.state_hidden_dim), nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.state_hidden_dim, self.config.state_hidden_dim),
            nn.GELU())
        self.lstm = nn.LSTM(
            self.config.visual_dim + self.config.state_hidden_dim,
            self.config.lstm_hidden_dim, self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.lstm_hidden_dim),
            nn.Linear(self.config.lstm_hidden_dim, 128), nn.GELU(),
            nn.Dropout(self.config.dropout), nn.Linear(128, 4))
        self.register_buffer(
            "command_scale",
            torch.tensor(self.config.command_scale, dtype=torch.float32),
            persistent=True)

    def initial_hidden(self, batch_size: int, device=None,
                       dtype=None) -> Tuple[Tensor, Tensor]:
        reference = next(self.parameters())
        device = device or reference.device
        dtype = dtype or reference.dtype
        shape = (self.config.lstm_layers, batch_size,
                 self.config.lstm_hidden_dim)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    def forward(self, depth: Tensor, state: Tensor,
                hidden: Optional[Tuple[Tensor, Tensor]] = None) -> PolicyOutput:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B,T,1,H,W]")
        if state.ndim != 3 or state.shape[:2] != depth.shape[:2] or \
                state.shape[-1] != self.config.state_dim:
            raise ValueError(
                "state must have shape [B,T,%d]" % self.config.state_dim)
        b, t = depth.shape[:2]
        frames = depth.reshape(b * t, 1, depth.shape[-2], depth.shape[-1])
        frames = F.interpolate(
            frames, size=(self.config.image_height, self.config.image_width),
            mode="bilinear", align_corners=False)
        visual = self.visual(frames).reshape(b, t, -1)
        visual = visual * self.config.visual_scale
        state_features = self.state_encoder(state.reshape(b * t, -1)).reshape(b, t, -1)
        recurrent, hidden_out = self.lstm(
            torch.cat((visual, state_features), dim=-1), hidden)
        normalized = torch.tanh(self.head(recurrent))
        return PolicyOutput(
            command=normalized * self.command_scale,
            normalized_command=normalized,
            hidden=hidden_out)

    def step(self, depth: Tensor, state: Tensor,
             hidden: Optional[Tuple[Tensor, Tensor]] = None) -> PolicyOutput:
        if depth.ndim != 4 or state.ndim != 2:
            raise ValueError("step expects depth [B,1,H,W] and state [B,11]")
        out = self.forward(depth[:, None], state[:, None], hidden)
        return PolicyOutput(out.command[:, 0], out.normalized_command[:, 0], out.hidden)


@dataclass
class MacroPolicyConfig:
    """Configuration for the causal 5 Hz upper-planner student.

    The macro student deliberately receives the ORIGINAL navigation goal
    (FLU direction + distance) and predicts the CORRECTED goal: unit FLU
    direction + normalized distance, pure regression with no PASS/NORMAL/
    TURN type and no direction token.  The predicted direction is
    world-latched by the runtime adapter (decision-time yaw) and re-projected
    into the 30 Hz student's current body frame.
    """

    image_height: int = 60
    image_width: int = 90
    # R31: 7-D macro state = gravity(3) + ORIGINAL goal(4); velocity/yaw_rate
    # removed so the 5 Hz policy must read depth, not short-circuit on motion.
    state_dim: int = 7
    stage_dims: Tuple[int, int] = (32, 64)
    stage_depths: Tuple[int, int] = (2, 2)
    stage_heads: Tuple[int, int] = (1, 2)
    sr_ratios: Tuple[int, int] = (8, 4)
    visual_dim: int = 256
    # Visual-feature scale alignment with the 30 Hz student (2026-09-02): the
    # 5 Hz macro policy now multiplies its visual embedding by the same 3.0
    # factor so the depth branch drives the recurrent state as strongly as
    # the state branch, mirroring the 30 Hz ViTFlyPolicy.
    visual_scale: float = 3.0
    state_hidden_dim: int = 96
    recurrent_hidden_dim: int = 192
    recurrent_layers: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        if self.state_dim != 7:
            raise ValueError("R31 macro state requires 7 inputs (gravity 3 + "
                             "original goal 4)")
        if self.image_height < 16 or self.image_width < 16:
            raise ValueError("image size is too small")
        if not (len(self.stage_dims) == len(self.stage_depths) ==
                len(self.stage_heads) == len(self.sr_ratios) == 2):
            raise ValueError("exactly two transformer stages are required")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MacroPolicyOutput:
    """Predictions for one or more 5 Hz macro decisions."""

    direction: Tensor
    distance_norm: Tensor
    hidden: Tuple[Tensor, Tensor]


class MacroPlannerPolicy(nn.Module):
    """Causal 5 Hz hierarchical-planner student (pure regression).

    Inputs are one depth frame and the 11-dimensional macro state
    ``gravity(3), velocity_flu(3), yaw_rate(1), original_goal(4)``.  The
    recurrent state is carried only at 5 Hz, so the six 30 Hz zero-order-held
    copies in a CSV episode are never treated as six independent decisions.
    Outputs are the CORRECTED target: unit FLU direction + normalized
    distance.  There is no PASS/NORMAL/TURN type and no direction token: a
    direction equal to the original goal is PASS, a deviation is NORMAL, and
    distance == 1.0 is the pure-rotation marker.  The runtime adapter
    world-latches the predicted direction (decision-time yaw) and re-projects
    it into the 30 Hz student's current body frame.
    """

    def __init__(self, config: Optional[MacroPolicyConfig] = None):
        super().__init__()
        self.config = config or MacroPolicyConfig()
        self.config.validate()
        visual_cfg = ViTFlyPolicyConfig(
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            stage_dims=self.config.stage_dims,
            stage_depths=self.config.stage_depths,
            stage_heads=self.config.stage_heads,
            sr_ratios=self.config.sr_ratios,
            visual_dim=self.config.visual_dim,
            dropout=self.config.dropout,
            visual_scale=self.config.visual_scale,
        )
        self.visual = VisualEncoder(visual_cfg)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.state_hidden_dim),
            nn.LayerNorm(self.config.state_hidden_dim), nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.state_hidden_dim,
                      self.config.state_hidden_dim), nn.GELU())
        fused_dim = self.config.visual_dim + self.config.state_hidden_dim
        self.recurrent = nn.LSTM(
            fused_dim, self.config.recurrent_hidden_dim,
            self.config.recurrent_layers, batch_first=True,
            dropout=self.config.dropout
            if self.config.recurrent_layers > 1 else 0.0)
        hidden = self.config.recurrent_hidden_dim
        self.norm = nn.LayerNorm(hidden)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(self.config.dropout), nn.Linear(hidden // 2, 3))
        self.distance_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(self.config.dropout), nn.Linear(hidden // 2, 1))

    def initial_hidden(self, batch_size: int, device=None,
                       dtype=None) -> Tuple[Tensor, Tensor]:
        reference = next(self.parameters())
        device = device or reference.device
        dtype = dtype or reference.dtype
        shape = (self.config.recurrent_layers, batch_size,
                 self.config.recurrent_hidden_dim)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    def forward(self, depth: Tensor, state: Tensor,
                hidden: Optional[Tuple[Tensor, Tensor]] = None
                ) -> MacroPolicyOutput:
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B,T,1,H,W]")
        if state.ndim != 3 or state.shape[:2] != depth.shape[:2] or \
                state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape [B,T,11]")
        b, t = depth.shape[:2]
        frames = depth.reshape(b * t, 1, depth.shape[-2], depth.shape[-1])
        frames = F.interpolate(
            frames, size=(self.config.image_height, self.config.image_width),
            mode="bilinear", align_corners=False)
        visual = self.visual(frames).reshape(b, t, -1)
        visual = visual * self.config.visual_scale
        state_features = self.state_encoder(
            state.reshape(b * t, -1)).reshape(b, t, -1)
        recurrent, hidden_out = self.recurrent(
            torch.cat((visual, state_features), dim=-1), hidden)
        features = self.norm(recurrent)
        direction = F.normalize(self.direction_head(features), dim=-1,
                                eps=1e-6)
        distance = torch.sigmoid(self.distance_head(features))
        return MacroPolicyOutput(
            direction=direction,
            distance_norm=distance,
            hidden=hidden_out)

    def step(self, depth: Tensor, state: Tensor,
             hidden: Optional[Tuple[Tensor, Tensor]] = None
             ) -> MacroPolicyOutput:
        if depth.ndim != 4 or state.ndim != 2:
            raise ValueError("step expects depth [B,1,H,W] and state [B,11]")
        out = self.forward(depth[:, None], state[:, None], hidden)
        return MacroPolicyOutput(
            direction=out.direction[:, 0],
            distance_norm=out.distance_norm[:, 0],
            hidden=out.hidden)

    @staticmethod
    def decode_directive(output: MacroPolicyOutput) -> Dict[str, Tensor]:
        """Return the raw corrected-target prediction (no type/token).

        The predicted FLU direction + normalized distance directly form the
        effective goal consumed by the 30 Hz student.  ``distance_norm`` is
        clamped to [0, 1] and keeps the goal-distance encoding shared with
        the 30 Hz input: [0, 0.9] ordinary tracking, 1.0 pure rotation.
        """
        return {"direction": output.direction,
                "distance_norm": output.distance_norm.squeeze(-1).clamp(0.0, 1.0)}


# Explicit names used by train.py and downstream deployment code.
PolicyConfig = ViTFlyPolicyConfig
Policy = ViTFlyLSTMPolicy
MacroPolicy = MacroPlannerPolicy
