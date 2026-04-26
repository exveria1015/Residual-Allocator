# coding: utf-8
# Copyright 2026 Exveria
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from models.bs_roformer.bs_roformer import TimeScreeningSelector
except Exception:
    TimeScreeningSelector = None


def upgrade_allocator_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    upgraded = dict(state_dict)
    legacy_prefix_map = {
        "context_norm.": "condition_context_norm.",
        "context_proj.": "condition_context_proj.",
    }
    for legacy_prefix, new_prefix in legacy_prefix_map.items():
        legacy_keys = [key for key in upgraded.keys() if key.startswith(legacy_prefix)]
        for key in legacy_keys:
            new_key = new_prefix + key[len(legacy_prefix):]
            if new_key not in upgraded:
                upgraded[new_key] = upgraded[key]
            upgraded.pop(key, None)
    return upgraded

class StemTimeScreeningRouter(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_bands: int = 64,
        num_frames: int = 128,
        eval_num_bands: int | None = None,
        eval_num_frames: int | None = None,
        token_dim: int = 32,
        heads: int = 4,
        dim_head: int = 16,
        dropout: float = 0.0,
        norm_values: bool = False,
        use_tanh_norm: bool = True,
        init_window: float = 64.0,
        init_relevance_width: float = 4.0,
        init_scale: float = 0.0,
        init_mode: str = "legacy",
        random_std: float = 0.02,
    ):
        super().__init__()
        if TimeScreeningSelector is None:
            raise ImportError("TimeScreeningSelector is unavailable; cannot use router_type='time_screening'")
        assert in_channels > 0, "in_channels must be positive"
        assert num_bands > 0, "num_bands must be positive"
        assert num_frames > 0, "num_frames must be positive"
        assert token_dim > 0, "token_dim must be positive"
        init_mode = str(init_mode).lower()
        assert init_mode in {"legacy", "random", "paper"}, "init_mode must be 'legacy', 'random', or 'paper'"
        assert random_std > 0.0, "random_std must be positive"
        if eval_num_bands is not None:
            assert eval_num_bands > 0, "eval_num_bands must be positive"
        if eval_num_frames is not None:
            assert eval_num_frames > 0, "eval_num_frames must be positive"

        self.num_bands = int(num_bands)
        self.num_frames = int(num_frames)
        self.eval_num_bands = None if eval_num_bands is None else int(eval_num_bands)
        self.eval_num_frames = None if eval_num_frames is None else int(eval_num_frames)
        self.init_mode = init_mode
        self.random_std = float(random_std)
        self.in_proj = nn.Linear(int(in_channels), int(token_dim), bias=False)
        self.screen = TimeScreeningSelector(
            dim=int(token_dim),
            heads=int(heads),
            dim_head=int(dim_head),
            dropout=float(dropout),
            rotary_embed=None,
            norm_values=bool(norm_values),
            use_tanh_norm=bool(use_tanh_norm),
            init_window=float(init_window),
            init_relevance_width=float(init_relevance_width),
            init_scale=float(init_scale),
        )
        self.to_logits = nn.Linear(int(token_dim), 1)
        self._reset_parameters(
            init_window=float(init_window),
            init_relevance_width=float(init_relevance_width),
            init_scale=float(init_scale),
        )

    def _reset_screen_scalars(
        self,
        *,
        init_window: float,
        init_relevance_width: float,
        init_scale: float,
    ) -> None:
        init_window = max(float(init_window), 1.0001)
        init_relevance_width = max(float(init_relevance_width), 1.0001)
        with torch.no_grad():
            self.screen.log_window.copy_(torch.log(torch.full_like(self.screen.log_window, init_window - 1.0)))
            self.screen.log_relevance_width.copy_(
                torch.log(torch.full_like(self.screen.log_relevance_width, init_relevance_width - 1.0))
            )
            self.screen.residual_scale.fill_(float(init_scale))

    def _reset_random_projection_weights(self, std: float) -> None:
        for module in (
            self.in_proj,
            self.screen.to_q,
            self.screen.to_k,
            self.screen.to_v,
            self.screen.to_gates,
            self.screen.to_out[0],
            self.to_logits,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=float(std))
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _apply_paper_init(self, init_window: float) -> None:
        qkv_std = 0.1 / math.sqrt(float(self.screen.to_q.out_features))
        model_std = 0.1 / math.sqrt(float(self.in_proj.out_features))
        gate_std = 0.1
        logit_std = 0.1 / math.sqrt(float(self.to_logits.in_features))

        nn.init.normal_(self.in_proj.weight, mean=0.0, std=model_std)
        nn.init.normal_(self.screen.to_q.weight, mean=0.0, std=qkv_std)
        nn.init.normal_(self.screen.to_k.weight, mean=0.0, std=qkv_std)
        nn.init.normal_(self.screen.to_v.weight, mean=0.0, std=qkv_std)
        nn.init.normal_(self.screen.to_gates.weight, mean=0.0, std=gate_std)
        nn.init.zeros_(self.screen.to_gates.bias)
        nn.init.normal_(self.screen.to_out[0].weight, mean=0.0, std=model_std)
        nn.init.normal_(self.to_logits.weight, mean=0.0, std=logit_std)
        nn.init.zeros_(self.to_logits.bias)

        w_th = max(float(init_window), 1.0001)
        paper_window_logs = torch.linspace(
            0.0,
            math.log(w_th),
            self.screen.log_window.numel(),
            dtype=self.screen.log_window.dtype,
            device=self.screen.log_window.device,
        )
        paper_window = torch.exp(paper_window_logs).clamp_min(1.0001)
        paper_window_param = torch.log(paper_window - 1.0)
        paper_residual_gain = min(1.0 / math.sqrt(float(self.screen.heads)), 0.999)
        paper_residual_scale = math.atanh(paper_residual_gain)
        with torch.no_grad():
            self.screen.log_window.copy_(paper_window_param)
            self.screen.log_relevance_width.zero_()
            self.screen.residual_scale.fill_(paper_residual_scale)

    def _reset_parameters(
        self,
        *,
        init_window: float,
        init_relevance_width: float,
        init_scale: float,
    ) -> None:
        if self.init_mode == "legacy":
            self._reset_screen_scalars(
                init_window=init_window,
                init_relevance_width=init_relevance_width,
                init_scale=init_scale,
            )
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.to_logits.bias)
            return

        if self.init_mode == "random":
            self._reset_screen_scalars(
                init_window=init_window,
                init_relevance_width=init_relevance_width,
                init_scale=init_scale,
            )
            self._reset_random_projection_weights(self.random_std)
            return

        self._apply_paper_init(init_window=init_window)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_stems, in_channels, freq_bins, time_steps = features.shape
        active_num_bands = self.num_bands if self.training or self.eval_num_bands is None else self.eval_num_bands
        active_num_frames = self.num_frames if self.training or self.eval_num_frames is None else self.eval_num_frames
        pooled = rearrange(features, "b n c f t -> (b n c) 1 f t")
        pooled_frames = min(active_num_frames, time_steps)
        pooled = F.adaptive_avg_pool2d(pooled, output_size=(active_num_bands, pooled_frames))
        pooled = rearrange(
            pooled,
            "(b n c) 1 band t -> (b n band) t c",
            b=batch_size,
            n=num_stems,
            c=in_channels,
            band=active_num_bands,
        )
        tokens = self.in_proj(pooled)
        screened = self.screen(tokens)
        logits = self.to_logits(screened)
        logits = rearrange(
            logits,
            "(b n band) t 1 -> b n band t",
            b=batch_size,
            n=num_stems,
            band=active_num_bands,
        )
        if active_num_bands != freq_bins or pooled_frames != time_steps:
            logits = F.interpolate(
                rearrange(logits, "b n band t -> (b n) 1 band t"),
                size=(freq_bins, time_steps),
                mode="bilinear",
                align_corners=False,
            )
            logits = rearrange(logits, "(b n) 1 f t -> b n f t", b=batch_size, n=num_stems)
        return logits


class CrossAttentionJudge(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        stem_summary_dim: int,
        latent_dim: int = 128,
        num_latents: int = 8,
        heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert context_dim > 0, "context_dim must be positive"
        assert stem_summary_dim > 0, "stem_summary_dim must be positive"
        assert latent_dim > 0, "latent_dim must be positive"
        assert num_latents > 0, "num_latents must be positive"
        assert heads > 0, "heads must be positive"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert latent_dim % heads == 0, "latent_dim must be divisible by heads"
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * (1.0 / math.sqrt(float(latent_dim))))
        self.query_norm = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.stem_norm = nn.LayerNorm(latent_dim + stem_summary_dim)
        self.stem_mlp = nn.Sequential(
            nn.Linear(latent_dim + stem_summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.stem_head = nn.Linear(hidden_dim, 3)
        self.global_head = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.stem_head.weight)
        nn.init.zeros_(self.stem_head.bias)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)

    def forward(self, context_tokens: torch.Tensor, stem_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        context_tokens = torch.nan_to_num(context_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        stem_summary = torch.nan_to_num(stem_summary.float(), nan=0.0, posinf=0.0, neginf=0.0)
        context_tokens = self.context_proj(self.context_norm(context_tokens))
        latents = self.latents.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        attended, _ = self.cross_attn(
            self.query_norm(latents),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        latents = latents + attended
        latents = latents + self.ff(self.ff_norm(latents))
        global_context = latents.mean(dim=1)
        stem_input = torch.cat(
            (
                global_context.unsqueeze(1).expand(-1, stem_summary.shape[1], -1),
                stem_summary,
            ),
            dim=-1,
        )
        stem_hidden = self.stem_mlp(self.stem_norm(stem_input))
        stem_outputs = self.stem_head(stem_hidden)
        return {
            "judge_score": torch.tanh(stem_outputs[..., 0]),
            "router_raw": stem_outputs[..., 1],
            "delta_raw": stem_outputs[..., 2],
            "blend_raw": self.global_head(global_context).squeeze(-1),
        }


class ConvResidualAllocator(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 16,
        kernel_size: int = 7,
        router_type: str = "conv",
        base_weight_gamma: float = 2.0,
        blend_init_bias: float = -4.0,
        blend_floor: float = 0.0,
        delta_scale: float = 4.0,
        residual_scale_init: float = 1.0,
        residual_scale_min: float = 0.25,
        residual_scale_max: float = 2.0,
        learn_residual_scale: bool = True,
        exclude_silent_stems: bool = False,
        silent_stem_abs_thresh: float = 0.0,
        silent_stem_rel_thresh: float = 0.0,
        silent_stem_time_kernel_size: int = 1,
        use_silent_reclaim: bool = False,
        silent_reclaim_strength: float = 1.0,
        silent_reclaim_soft_width: float = 0.25,
        use_exact_mix_closure: bool = False,
        exact_mix_closure_topk: int = 0,
        screening_num_bands: int = 64,
        screening_num_frames: int = 128,
        screening_eval_num_bands: int | None = None,
        screening_eval_num_frames: int | None = None,
        screening_token_dim: int = 32,
        screening_heads: int = 4,
        screening_dim_head: int = 16,
        screening_dropout: float = 0.0,
        screening_norm_values: bool = False,
        screening_tanh_norm: bool = True,
        screening_init_window: float = 64.0,
        screening_init_relevance_width: float = 4.0,
        screening_init_scale: float = 0.0,
        screening_init_mode: str = "legacy",
        screening_random_std: float = 0.02,
        use_small_delta_branch: bool = False,
        delta_branch_hidden_channels: int | None = None,
        delta_branch_freq_kernel_size: int = 5,
        delta_branch_time_kernel_size: int = 3,
        delta_branch_scale: float = 0.1,
        use_context_conditioning: bool = False,
        context_feature_dim: int = 0,
        condition_mode: str = "mlp",
        condition_hidden_dim: int = 128,
        condition_router_scale: float = 0.5,
        condition_delta_scale: float = 0.5,
        condition_blend_scale: float = 0.25,
        judge_num_latents: int = 8,
        judge_latent_dim: int = 128,
        judge_heads: int = 4,
        judge_context_num_frames: int = 16,
        judge_context_num_bands: int = 8,
        judge_dropout: float = 0.0,
        inactive_conf_scale: float = 0.2,
        inactive_keep_floor: float = 0.25,
        inactive_keep_max: float = 1.0,
        use_artifact_detector: bool = False,
        artifact_hidden_channels: int | None = None,
        artifact_kernel_size: int = 3,
        artifact_init_bias: float = -6.0,
        artifact_max_suppression: float = 1.0,
        artifact_keep_floor: float = 0.0,
        artifact_base_active_mix_ratio: float = 0.02,
        artifact_gt_inactive_mix_ratio: float = 0.006,
        artifact_gt_active_mix_ratio: float = 0.02,
        artifact_over_gt_margin_db: float = 12.0,
    ):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be a positive odd integer"
        assert delta_branch_freq_kernel_size >= 1 and delta_branch_freq_kernel_size % 2 == 1, (
            "delta_branch_freq_kernel_size must be a positive odd integer"
        )
        assert delta_branch_time_kernel_size >= 1 and delta_branch_time_kernel_size % 2 == 1, (
            "delta_branch_time_kernel_size must be a positive odd integer"
        )
        assert 0.0 <= blend_floor < 1.0, "blend_floor must be in [0, 1)"
        assert delta_scale > 0.0, "delta_scale must be positive"
        assert delta_branch_scale >= 0.0, "delta_branch_scale must be non-negative"
        assert context_feature_dim >= 0, "context_feature_dim must be non-negative"
        condition_mode = str(condition_mode).lower()
        assert condition_mode in {"mlp", "cross_attention"}, "condition_mode must be 'mlp' or 'cross_attention'"
        assert condition_hidden_dim > 0, "condition_hidden_dim must be positive"
        assert condition_router_scale >= 0.0, "condition_router_scale must be non-negative"
        assert condition_delta_scale >= 0.0, "condition_delta_scale must be non-negative"
        assert condition_blend_scale >= 0.0, "condition_blend_scale must be non-negative"
        assert judge_num_latents > 0, "judge_num_latents must be positive"
        assert judge_latent_dim > 0, "judge_latent_dim must be positive"
        assert judge_heads > 0, "judge_heads must be positive"
        assert judge_context_num_frames > 0, "judge_context_num_frames must be positive"
        assert judge_context_num_bands > 0, "judge_context_num_bands must be positive"
        assert 0.0 <= inactive_conf_scale <= 1.0, "inactive_conf_scale must be in [0, 1]"
        assert 0.0 <= inactive_keep_floor <= 1.0, "inactive_keep_floor must be in [0, 1]"
        assert 0.0 <= inactive_keep_max <= 1.0, "inactive_keep_max must be in [0, 1]"
        assert artifact_kernel_size >= 1 and artifact_kernel_size % 2 == 1, (
            "artifact_kernel_size must be a positive odd integer"
        )
        assert 0.0 <= artifact_max_suppression <= 1.0, "artifact_max_suppression must be in [0, 1]"
        assert 0.0 <= artifact_keep_floor <= 1.0, "artifact_keep_floor must be in [0, 1]"
        assert artifact_base_active_mix_ratio >= 0.0, "artifact_base_active_mix_ratio must be non-negative"
        assert artifact_gt_inactive_mix_ratio >= 0.0, "artifact_gt_inactive_mix_ratio must be non-negative"
        assert artifact_gt_active_mix_ratio >= 0.0, "artifact_gt_active_mix_ratio must be non-negative"
        assert residual_scale_min > 0.0, "residual_scale_min must be positive"
        assert residual_scale_max >= residual_scale_min, "residual_scale_max must be >= residual_scale_min"
        assert silent_stem_abs_thresh >= 0.0, "silent_stem_abs_thresh must be non-negative"
        assert silent_stem_rel_thresh >= 0.0, "silent_stem_rel_thresh must be non-negative"
        assert silent_stem_time_kernel_size > 0 and silent_stem_time_kernel_size % 2 == 1, (
            "silent_stem_time_kernel_size must be a positive odd integer"
        )
        assert silent_reclaim_strength >= 0.0, "silent_reclaim_strength must be non-negative"
        assert silent_reclaim_soft_width > 0.0, "silent_reclaim_soft_width must be positive"
        assert exact_mix_closure_topk >= 0, "exact_mix_closure_topk must be non-negative"
        router_type = str(router_type).lower()
        assert router_type in {"conv", "time_screening"}, "router_type must be 'conv' or 'time_screening'"
        self.base_weight_gamma = float(base_weight_gamma)
        self.blend_floor = float(blend_floor)
        self.delta_scale = float(delta_scale)
        self.residual_scale_min = float(residual_scale_min)
        self.residual_scale_max = float(residual_scale_max)
        self.learn_residual_scale = bool(learn_residual_scale)
        self.exclude_silent_stems = bool(exclude_silent_stems)
        self.silent_stem_abs_thresh = float(silent_stem_abs_thresh)
        self.silent_stem_rel_thresh = float(silent_stem_rel_thresh)
        self.silent_stem_time_kernel_size = int(silent_stem_time_kernel_size)
        self.use_silent_reclaim = bool(use_silent_reclaim)
        self.silent_reclaim_strength = float(silent_reclaim_strength)
        self.silent_reclaim_soft_width = float(silent_reclaim_soft_width)
        self.use_exact_mix_closure = bool(use_exact_mix_closure)
        self.exact_mix_closure_topk = int(exact_mix_closure_topk)
        self.router_type = router_type
        self.use_small_delta_branch = bool(use_small_delta_branch)
        self.delta_branch_scale = float(delta_branch_scale)
        self.use_context_conditioning = bool(use_context_conditioning)
        self.context_feature_dim = int(context_feature_dim)
        self.condition_mode = condition_mode
        self.condition_hidden_dim = int(condition_hidden_dim)
        self.condition_router_scale = float(condition_router_scale)
        self.condition_delta_scale = float(condition_delta_scale)
        self.condition_blend_scale = float(condition_blend_scale)
        self.judge_context_num_frames = int(judge_context_num_frames)
        self.judge_context_num_bands = int(judge_context_num_bands)
        self.inactive_conf_scale = float(inactive_conf_scale)
        self.inactive_keep_floor = float(inactive_keep_floor)
        self.inactive_keep_max = float(inactive_keep_max)
        self.use_artifact_detector = bool(use_artifact_detector)
        self.artifact_max_suppression = float(artifact_max_suppression)
        self.artifact_keep_floor = float(artifact_keep_floor)
        self.artifact_base_active_mix_ratio = float(artifact_base_active_mix_ratio)
        self.artifact_gt_inactive_mix_ratio = float(artifact_gt_inactive_mix_ratio)
        self.artifact_gt_active_mix_ratio = float(artifact_gt_active_mix_ratio)
        self.artifact_over_gt_margin_db = float(artifact_over_gt_margin_db)
        self.stem_summary_dim = 10
        if self.router_type == "conv":
            padding = (kernel_size // 2, 0)
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=(kernel_size, 1), padding=padding),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
            final_conv = self.net[-1]
            nn.init.normal_(final_conv.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(final_conv.bias)
        else:
            self.net = StemTimeScreeningRouter(
                in_channels=in_channels,
                num_bands=int(screening_num_bands),
                num_frames=int(screening_num_frames),
                eval_num_bands=screening_eval_num_bands,
                eval_num_frames=screening_eval_num_frames,
                token_dim=int(screening_token_dim),
                heads=int(screening_heads),
                dim_head=int(screening_dim_head),
                dropout=float(screening_dropout),
                norm_values=bool(screening_norm_values),
                use_tanh_norm=bool(screening_tanh_norm),
                init_window=float(screening_init_window),
                init_relevance_width=float(screening_init_relevance_width),
                init_scale=float(screening_init_scale),
                init_mode=str(screening_init_mode),
                random_std=float(screening_random_std),
            )
        if self.use_small_delta_branch:
            delta_hidden_channels = int(hidden_channels if delta_branch_hidden_channels is None else delta_branch_hidden_channels)
            delta_padding = (
                delta_branch_freq_kernel_size // 2,
                delta_branch_time_kernel_size // 2,
            )
            self.delta_net = nn.Sequential(
                nn.Conv2d(
                    in_channels + 4,
                    delta_hidden_channels,
                    kernel_size=(delta_branch_freq_kernel_size, delta_branch_time_kernel_size),
                    padding=delta_padding,
                ),
                nn.SiLU(),
                nn.Conv2d(delta_hidden_channels, 2, kernel_size=1),
            )
            delta_final_conv = self.delta_net[-1]
            nn.init.zeros_(delta_final_conv.weight)
            nn.init.zeros_(delta_final_conv.bias)
        else:
            self.delta_net = None
        if self.use_artifact_detector:
            artifact_hidden = int(hidden_channels if artifact_hidden_channels is None else artifact_hidden_channels)
            artifact_padding = artifact_kernel_size // 2
            self.artifact_net = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    artifact_hidden,
                    kernel_size=(artifact_kernel_size, artifact_kernel_size),
                    padding=(artifact_padding, artifact_padding),
                ),
                nn.SiLU(),
                nn.Conv2d(artifact_hidden, 1, kernel_size=1),
            )
            artifact_final_conv = self.artifact_net[-1]
            nn.init.zeros_(artifact_final_conv.weight)
            nn.init.constant_(artifact_final_conv.bias, float(artifact_init_bias))
        else:
            self.artifact_net = None
        if self.use_context_conditioning:
            assert self.context_feature_dim > 0, "context_feature_dim must be positive when context conditioning is enabled"
            if self.condition_mode == "mlp":
                self.condition_context_norm = nn.LayerNorm(self.context_feature_dim)
                self.condition_context_proj = nn.Sequential(
                    nn.Linear(self.context_feature_dim, self.condition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.condition_hidden_dim, self.condition_hidden_dim),
                    nn.SiLU(),
                )
                self.condition_net = nn.Sequential(
                    nn.Linear(self.condition_hidden_dim + self.stem_summary_dim, self.condition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.condition_hidden_dim, self.condition_hidden_dim),
                    nn.SiLU(),
                )
                self.condition_head = nn.Linear(self.condition_hidden_dim, 3)
                self.global_condition_head = nn.Linear(self.condition_hidden_dim, 1)
                nn.init.zeros_(self.condition_head.weight)
                nn.init.zeros_(self.condition_head.bias)
                nn.init.zeros_(self.global_condition_head.weight)
                nn.init.zeros_(self.global_condition_head.bias)
                self.cross_attention_judge = None
            else:
                self.condition_context_norm = None
                self.condition_context_proj = None
                self.condition_net = None
                self.condition_head = None
                self.global_condition_head = None
                self.cross_attention_judge = CrossAttentionJudge(
                    context_dim=self.context_feature_dim,
                    stem_summary_dim=self.stem_summary_dim,
                    latent_dim=int(judge_latent_dim),
                    num_latents=int(judge_num_latents),
                    heads=int(judge_heads),
                    hidden_dim=self.condition_hidden_dim,
                    dropout=float(judge_dropout),
                )
        else:
            self.condition_context_norm = None
            self.condition_context_proj = None
            self.condition_net = None
            self.condition_head = None
            self.global_condition_head = None
            self.cross_attention_judge = None
        self.blend_logit = nn.Parameter(torch.tensor(float(blend_init_bias), dtype=torch.float32))
        if self.learn_residual_scale:
            self.log_residual_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale_init), dtype=torch.float32)))
        else:
            self.register_buffer(
                "fixed_residual_scale",
                torch.tensor(float(residual_scale_init), dtype=torch.float32),
                persistent=True,
            )

    def _audio_to_flattened_stft(self, base_model, audio: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not hasattr(base_model, "_audio_to_flattened_stft"):
            raise AttributeError("base_model must define _audio_to_flattened_stft()")
        return base_model._audio_to_flattened_stft(audio, device)

    def _flattened_stft_to_audio(
        self,
        base_model,
        masked_stft: torch.Tensor,
        device: torch.device,
        audio_length: int,
    ) -> torch.Tensor:
        if not hasattr(base_model, "_flattened_stft_to_audio"):
            raise AttributeError("base_model must define _flattened_stft_to_audio()")
        return base_model._flattened_stft_to_audio(masked_stft, device, audio_length)

    def _apply_waveform_mix_closure(
        self,
        recon_audio: torch.Tensor,
        mix_audio: torch.Tensor,
        *,
        prior: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mix_audio = mix_audio[..., :recon_audio.shape[-1]]
        residual = mix_audio.unsqueeze(1) - recon_audio.sum(dim=1, keepdim=True)

        weights = recon_audio.abs()
        fallback_weights = None
        if prior is not None:
            prior = prior.to(device=weights.device, dtype=weights.dtype).clamp_min(0.0)
            if prior.ndim == 2:
                prior = rearrange(prior, "b n -> b n 1 1")
            elif prior.ndim == 3:
                prior = rearrange(prior, "b n t -> b n 1 t")
            elif prior.ndim != 4:
                raise ValueError(f"Unsupported prior rank for mix closure: {prior.shape}")
            fallback_weights = prior
            weights = weights * prior
        if candidate_mask is not None:
            candidate_mask = self._resize_time_mask(
                candidate_mask.to(device=weights.device),
                weights.shape[-1],
            )
            masked_weights = weights * candidate_mask.to(dtype=weights.dtype)
            masked_weights_sum = masked_weights.sum(dim=1, keepdim=True)
            weights = torch.where(masked_weights_sum > eps, masked_weights, weights)

        if 0 < self.exact_mix_closure_topk < weights.shape[1]:
            topk_indices = weights.topk(k=self.exact_mix_closure_topk, dim=1).indices
            keep_mask = torch.zeros_like(weights)
            keep_mask.scatter_(1, topk_indices, 1.0)
            weights = weights * keep_mask
            if fallback_weights is not None:
                fallback_keep_mask = torch.zeros_like(fallback_weights)
                fallback_topk_indices = fallback_weights.topk(k=self.exact_mix_closure_topk, dim=1).indices
                fallback_keep_mask.scatter_(1, fallback_topk_indices, 1.0)
                fallback_weights = fallback_weights * fallback_keep_mask

        weights_sum = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / recon_audio.shape[1])
        if fallback_weights is not None:
            fallback_sum = fallback_weights.sum(dim=1, keepdim=True)
            fallback_weights = torch.where(
                fallback_sum > eps,
                fallback_weights / fallback_sum.clamp_min(eps),
                uniform,
            )
        else:
            fallback_weights = uniform
        weights = torch.where(weights_sum > eps, weights / weights_sum.clamp_min(eps), fallback_weights)

        corrected = recon_audio + weights * residual
        closure_residual = mix_audio - corrected.sum(dim=1)
        dominant_stem = corrected.abs().argmax(dim=1, keepdim=True)
        corrected = corrected.scatter_add(1, dominant_stem, closure_residual.unsqueeze(1))
        return corrected, weights, closure_residual

    def _resize_time_mask(
        self,
        mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        if mask.ndim == 3:
            batch_size, num_stems = mask.shape[:2]
            flat_mask = rearrange(mask.to(dtype=torch.float32), "b n t -> (b n) 1 t")
        elif mask.ndim == 4 and mask.shape[2] == 1:
            batch_size, num_stems = mask.shape[:2]
            flat_mask = rearrange(mask.to(dtype=torch.float32), "b n 1 t -> (b n) 1 t")
        else:
            raise ValueError(f"Unsupported time mask rank: {mask.shape}")

        if flat_mask.shape[-1] != target_length:
            flat_mask = F.interpolate(flat_mask, size=target_length, mode="nearest")

        return rearrange(
            flat_mask > 0.5,
            "(b n) 1 t -> b n 1 t",
            b=batch_size,
            n=num_stems,
        )

    def _resize_time_gate(
        self,
        gate: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        if gate.ndim == 3:
            batch_size, num_stems = gate.shape[:2]
            flat_gate = rearrange(gate.to(dtype=torch.float32), "b n t -> (b n) 1 t")
        elif gate.ndim == 4 and gate.shape[2] == 1:
            batch_size, num_stems = gate.shape[:2]
            flat_gate = rearrange(gate.to(dtype=torch.float32), "b n 1 t -> (b n) 1 t")
        else:
            raise ValueError(f"Unsupported time gate rank: {gate.shape}")

        if flat_gate.shape[-1] != target_length:
            flat_gate = F.interpolate(flat_gate, size=target_length, mode="linear", align_corners=False)

        flat_gate = flat_gate.clamp_(0.0, 1.0)
        return rearrange(
            flat_gate,
            "(b n) 1 t -> b n 1 t",
            b=batch_size,
            n=num_stems,
        )

    def _merge_stem_activity_masks(
        self,
        *,
        built_mask: torch.Tensor | None,
        override_mask: torch.Tensor | None,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if override_mask is None:
            return built_mask

        override_mask = self._resize_time_mask(
            override_mask.to(device=device),
            target_length,
        )
        if built_mask is None:
            return override_mask
        return built_mask & override_mask

    def _build_stem_activity_mask(
        self,
        *,
        stem_mag: torch.Tensor,
        mix_mag: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.exclude_silent_stems:
            return None

        batch_size, num_stems = stem_mag.shape[:2]
        stem_activity = stem_mag.mean(dim=2, keepdim=True)
        if self.silent_stem_time_kernel_size > 1:
            stem_activity = F.max_pool1d(
                rearrange(stem_activity, "b n 1 t -> (b n) 1 t"),
                kernel_size=self.silent_stem_time_kernel_size,
                stride=1,
                padding=self.silent_stem_time_kernel_size // 2,
            )
            stem_activity = rearrange(
                stem_activity,
                "(b n) 1 t -> b n 1 t",
                b=batch_size,
                n=num_stems,
            )
        mix_activity = mix_mag.mean(dim=2, keepdim=True)
        threshold = stem_activity.new_tensor(self.silent_stem_abs_thresh)
        if self.silent_stem_rel_thresh > 0.0:
            threshold = torch.maximum(threshold, mix_activity * self.silent_stem_rel_thresh)
        active_mask = stem_activity > threshold
        return active_mask

    def _build_stem_silent_reclaim_gate(
        self,
        *,
        stem_mag: torch.Tensor,
        mix_mag: torch.Tensor,
        eps: float,
    ) -> torch.Tensor | None:
        if not self.use_silent_reclaim:
            return None

        batch_size, num_stems = stem_mag.shape[:2]
        stem_activity = stem_mag.mean(dim=2, keepdim=True)
        if self.silent_stem_time_kernel_size > 1:
            stem_activity = F.max_pool1d(
                rearrange(stem_activity, "b n 1 t -> (b n) 1 t"),
                kernel_size=self.silent_stem_time_kernel_size,
                stride=1,
                padding=self.silent_stem_time_kernel_size // 2,
            )
            stem_activity = rearrange(
                stem_activity,
                "(b n) 1 t -> b n 1 t",
                b=batch_size,
                n=num_stems,
            )

        mix_activity = mix_mag.mean(dim=2, keepdim=True)
        threshold = stem_activity.new_tensor(self.silent_stem_abs_thresh)
        if self.silent_stem_rel_thresh > 0.0:
            threshold = torch.maximum(threshold, mix_activity * self.silent_stem_rel_thresh)

        width = threshold * self.silent_reclaim_soft_width + eps
        reclaim_gate = torch.sigmoid((threshold - stem_activity) / width)
        reclaim_gate = reclaim_gate * self.silent_reclaim_strength
        reclaim_gate = torch.clamp(reclaim_gate, min=0.0, max=1.0)
        return reclaim_gate

    def _build_stem_summary_features(
        self,
        *,
        owned_raw: torch.Tensor,
        peakiness: torch.Tensor,
        residual_ratio: torch.Tensor,
        stem_to_mix: torch.Tensor,
        phase_agreement: torch.Tensor,
        ownership_margin: torch.Tensor,
        stem_mag: torch.Tensor,
        mix_mag: torch.Tensor,
        residual_mag: torch.Tensor,
        base_weights: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        stem_energy = stem_mag.mean(dim=(2, 3))
        mix_energy = mix_mag.mean(dim=(2, 3)).expand_as(stem_energy)
        residual_energy = residual_mag.mean(dim=(2, 3)).expand_as(stem_energy)
        summary = torch.stack(
            (
                owned_raw.mean(dim=(2, 3)),
                peakiness.mean(dim=(2, 3)),
                residual_ratio.mean(dim=(2, 3)).expand_as(stem_energy),
                stem_to_mix.mean(dim=(2, 3)),
                phase_agreement.mean(dim=(2, 3)),
                ownership_margin.mean(dim=(2, 3)),
                base_weights.mean(dim=(2, 3)),
                torch.log1p(stem_mag).mean(dim=(2, 3)),
                torch.log1p(stem_energy),
                torch.log1p(residual_energy / mix_energy.clamp_min(eps)),
            ),
            dim=-1,
        )
        return torch.nan_to_num(summary.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_context_condition(
        self,
        *,
        base_model,
        mix_audio: torch.Tensor,
        stem_summary: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor] | None:
        if not self.use_context_conditioning:
            return None
        if self.condition_mode == "cross_attention":
            if not hasattr(base_model, "extract_refiner_condition_tokens"):
                raise AttributeError(
                    "base_model must define extract_refiner_condition_tokens() when condition_mode='cross_attention'"
                )
            with torch.no_grad():
                context_tokens = base_model.extract_refiner_condition_tokens(
                    mix_audio,
                    device=device,
                    num_frames=self.judge_context_num_frames,
                    num_bands=self.judge_context_num_bands,
                )
            condition_raw = self.cross_attention_judge(
                context_tokens.to(device=device, dtype=dtype),
                stem_summary.to(device=device, dtype=dtype),
            )
        else:
            if not hasattr(base_model, "extract_refiner_condition_vector"):
                raise AttributeError(
                    "base_model must define extract_refiner_condition_vector() when use_context_conditioning=True"
                )
            with torch.no_grad():
                context_vector = base_model.extract_refiner_condition_vector(mix_audio, device=device)
            context_vector = torch.nan_to_num(
                context_vector.to(device=device, dtype=dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            context_hidden = self.condition_context_proj(self.condition_context_norm(context_vector))
            condition_input = torch.cat(
                (
                    context_hidden.unsqueeze(1).expand(-1, stem_summary.shape[1], -1),
                    stem_summary.to(device=device, dtype=dtype),
                ),
                dim=-1,
            )
            stem_hidden = self.condition_net(condition_input)
            stem_outputs = self.condition_head(stem_hidden)
            condition_raw = {
                "judge_score": torch.tanh(stem_outputs[..., 0]),
                "router_raw": stem_outputs[..., 1],
                "delta_raw": stem_outputs[..., 2],
                "blend_raw": self.global_condition_head(context_hidden).squeeze(-1),
            }

        judge_score = condition_raw["judge_score"]
        router_bias = self.condition_router_scale * torch.tanh(condition_raw["router_raw"])
        delta_gain = (1.0 + self.condition_delta_scale * torch.tanh(condition_raw["delta_raw"])).clamp_min(0.0)
        blend_delta = self.condition_blend_scale * torch.tanh(condition_raw["blend_raw"])
        return {
            "judge_score": judge_score,
            "router_bias": router_bias,
            "delta_gain": delta_gain,
            "blend_delta": blend_delta,
        }

    def _build_judge_teacher(
        self,
        *,
        base_model,
        target_audio: torch.Tensor,
        stem_mag: torch.Tensor,
        mix_mag: torch.Tensor,
        device: torch.device,
        eps: float,
    ) -> torch.Tensor:
        target_stft = self._audio_to_flattened_stft(base_model, target_audio, device)
        target_mag = target_stft.abs()
        scale = mix_mag.mean(dim=(2, 3)).expand_as(stem_mag.mean(dim=(2, 3))).clamp_min(eps)
        under_amount = F.relu(target_mag - stem_mag).mean(dim=(2, 3)) / scale
        over_amount = F.relu(stem_mag - target_mag).mean(dim=(2, 3)) / scale
        return torch.tanh(under_amount - over_amount)

    def _build_artifact_features(
        self,
        *,
        stem_stft: torch.Tensor,
        mix_stft: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        stem_mag = stem_stft.abs()
        mix_mag = mix_stft.abs().clamp_min(eps)
        residual_stft = mix_stft - stem_stft.sum(dim=1, keepdim=True)
        residual_mag = residual_stft.abs()

        owned_raw = stem_mag / stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
        peakiness = owned_raw / owned_raw.amax(dim=1, keepdim=True).clamp_min(eps)
        residual_ratio = residual_mag / mix_mag
        stem_to_mix = stem_mag / mix_mag
        phase_agreement = (
            (stem_stft.real * residual_stft.real) + (stem_stft.imag * residual_stft.imag)
        ) / (stem_mag * residual_mag + eps)
        phase_agreement = torch.clamp(phase_agreement, min=-1.0, max=1.0)

        top2 = torch.topk(owned_raw, k=min(2, owned_raw.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            ownership_margin = top2[:, :1]
        else:
            ownership_margin = top2[:, :1] - top2[:, 1:2]
        ownership_margin = ownership_margin.expand_as(owned_raw)

        features = torch.stack(
            (
                owned_raw,
                peakiness,
                residual_ratio.expand_as(owned_raw),
                stem_to_mix,
                phase_agreement,
                ownership_margin,
                torch.log1p(stem_mag),
            ),
            dim=2,
        )
        return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _build_artifact_teacher(
        self,
        *,
        base_model,
        target_audio: torch.Tensor,
        pre_detector_stft: torch.Tensor,
        mix_stft: torch.Tensor,
        device: torch.device,
        eps: float,
    ) -> dict[str, torch.Tensor]:
        target_stft = self._audio_to_flattened_stft(base_model, target_audio, device)
        base_mag = pre_detector_stft.abs()
        target_mag = target_stft.abs()
        mix_mag = mix_stft.abs().clamp_min(eps)

        base_to_mix = base_mag / mix_mag
        target_to_mix = target_mag / mix_mag
        margin = float(10.0 ** (self.artifact_over_gt_margin_db / 20.0))

        base_active = base_to_mix > self.artifact_base_active_mix_ratio
        gt_inactive = target_to_mix < self.artifact_gt_inactive_mix_ratio
        gt_active = target_to_mix > self.artifact_gt_active_mix_ratio
        over_gt = base_to_mix > (target_to_mix * margin + self.artifact_base_active_mix_ratio)

        drop_target = (base_active & gt_inactive & over_gt).to(dtype=base_mag.dtype)
        protect_target = gt_active.to(dtype=base_mag.dtype)
        loss_mask = (base_active | gt_active).to(dtype=base_mag.dtype)
        return {
            "artifact_teacher": drop_target,
            "artifact_loss_mask": loss_mask,
            "artifact_protect_mask": protect_target,
            "artifact_base_active_mask": base_active.to(dtype=base_mag.dtype),
        }

    def forward(
        self,
        base_model,
        mix_audio: torch.Tensor,
        base_pred_audio: torch.Tensor,
        device: torch.device,
        target_audio: torch.Tensor | None = None,
        stem_activity_mask_override: torch.Tensor | None = None,
        stem_reclaim_mask_override: torch.Tensor | None = None,
        stem_delta_gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eps = 1e-8
        stem_stft = self._audio_to_flattened_stft(base_model, base_pred_audio, device)
        mix_stft = self._audio_to_flattened_stft(base_model, mix_audio, device)
        pre_reclaim_stem_mag = stem_stft.abs()
        pre_reclaim_mix_mag = mix_stft.abs().clamp_min(eps)
        pre_owned_raw = pre_reclaim_stem_mag / pre_reclaim_stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
        pre_top2 = torch.topk(pre_owned_raw, k=min(2, pre_owned_raw.shape[1]), dim=1).values
        if pre_top2.shape[1] == 1:
            pre_ownership_margin = pre_top2[:, :1]
        else:
            pre_ownership_margin = pre_top2[:, :1] - pre_top2[:, 1:2]
        pre_ownership_margin = pre_ownership_margin.expand_as(pre_owned_raw)
        pre_stem_to_mix = pre_reclaim_stem_mag / pre_reclaim_mix_mag

        reclaim_keep_gate = None
        gate_confidence = None
        activity_gate = None
        silent_reclaim_gate = None
        reclaimed_stft = torch.zeros_like(stem_stft)
        if stem_reclaim_mask_override is not None or stem_delta_gate_override is not None:
            ownership_keep = torch.sigmoid((pre_ownership_margin - 0.08) * 10.0)
            mix_ratio_keep = torch.sigmoid((pre_stem_to_mix - 0.04) * 12.0)
            gate_confidence = (ownership_keep * mix_ratio_keep).mean(dim=2, keepdim=True)
            gate_confidence = torch.nan_to_num(gate_confidence.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        if gate_confidence is not None and stem_activity_mask_override is not None:
            activity_gate = self._resize_time_gate(
                stem_activity_mask_override.to(device=stem_stft.device, dtype=stem_stft.real.dtype),
                stem_stft.shape[-1],
            )
            inactive_gate_scale = torch.where(
                activity_gate > 0.5,
                torch.ones_like(activity_gate),
                torch.full_like(activity_gate, self.inactive_conf_scale),
            )
            gate_confidence = gate_confidence.to(device=stem_stft.device, dtype=stem_stft.real.dtype)
            gate_confidence = (gate_confidence * inactive_gate_scale).clamp_(0.0, 1.0)
        if stem_reclaim_mask_override is not None:
            reclaim_keep_gate = self._resize_time_gate(
                stem_reclaim_mask_override.to(device=stem_stft.device),
                stem_stft.shape[-1],
            )
            reclaim_keep_gate = reclaim_keep_gate.to(device=stem_stft.device, dtype=stem_stft.real.dtype)
            if gate_confidence is not None:
                gate_confidence = gate_confidence.to(device=stem_stft.device, dtype=stem_stft.real.dtype)
                reclaim_strength = (1.0 - reclaim_keep_gate) * (1.0 - gate_confidence)
                reclaim_keep_gate = (1.0 - reclaim_strength).clamp_(0.0, 1.0)
                if activity_gate is not None:
                    inactive_keep_ceiling = self.inactive_keep_floor + (1.0 - self.inactive_keep_floor) * gate_confidence
                    if self.inactive_keep_max < 1.0:
                        inactive_keep_ceiling = torch.minimum(
                            inactive_keep_ceiling,
                            torch.full_like(inactive_keep_ceiling, self.inactive_keep_max),
                        )
                    reclaim_keep_gate = torch.where(
                        activity_gate > 0.5,
                        reclaim_keep_gate,
                        torch.minimum(reclaim_keep_gate, inactive_keep_ceiling),
                    )
            reclaimed_stft = stem_stft * (1.0 - reclaim_keep_gate)
            stem_stft = stem_stft * reclaim_keep_gate
        elif self.use_silent_reclaim:
            stem_mag = stem_stft.abs()
            mix_mag = mix_stft.abs().clamp_min(eps)
            silent_reclaim_gate = self._build_stem_silent_reclaim_gate(
                stem_mag=stem_mag,
                mix_mag=mix_mag,
                eps=eps,
            )
            reclaimed_stft = rearrange(
                silent_reclaim_gate.to(device=stem_stft.device, dtype=stem_stft.real.dtype),
                "b n 1 t -> b n 1 t",
            ) * stem_stft
            stem_stft = stem_stft - reclaimed_stft.to(stem_stft.dtype)

        artifact_logits = None
        artifact_prob = None
        artifact_keep_gate = None
        artifact_removed_stft = torch.zeros_like(stem_stft)
        pre_artifact_stft = stem_stft
        if self.artifact_net is not None:
            artifact_features = self._build_artifact_features(
                stem_stft=stem_stft,
                mix_stft=mix_stft,
                eps=eps,
            )
            batch_size, num_stems = artifact_features.shape[:2]
            artifact_logits = self.artifact_net(
                rearrange(artifact_features, "b n c f t -> (b n) c f t")
            )
            artifact_logits = rearrange(
                artifact_logits,
                "(b n) 1 f t -> b n f t",
                b=batch_size,
                n=num_stems,
            )
            artifact_logits = torch.nan_to_num(artifact_logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
            artifact_prob = torch.sigmoid(artifact_logits)
            artifact_keep_gate = 1.0 - self.artifact_max_suppression * artifact_prob
            if self.artifact_keep_floor > 0.0:
                artifact_keep_gate = artifact_keep_gate.clamp_min(self.artifact_keep_floor)
            artifact_keep_gate = artifact_keep_gate.clamp_(0.0, 1.0)
            artifact_removed_stft = stem_stft * (1.0 - artifact_keep_gate.to(dtype=stem_stft.real.dtype))
            stem_stft = stem_stft * artifact_keep_gate.to(dtype=stem_stft.real.dtype)

        stem_mag = stem_stft.abs()
        mix_mag = mix_stft.abs().clamp_min(eps)
        residual_stft = mix_stft - stem_stft.sum(dim=1, keepdim=True)
        residual_mag = residual_stft.abs()
        stem_activity_mask = self._build_stem_activity_mask(
            stem_mag=stem_mag,
            mix_mag=mix_mag,
        )
        stem_activity_mask = self._merge_stem_activity_masks(
            built_mask=stem_activity_mask,
            override_mask=stem_activity_mask_override,
            target_length=stem_mag.shape[-1],
            device=stem_mag.device,
        )
        delta_activity_gate = None
        if stem_delta_gate_override is not None:
            delta_activity_gate = self._resize_time_gate(
                stem_delta_gate_override.to(device=stem_mag.device),
                stem_mag.shape[-1],
            ).to(device=stem_mag.device, dtype=stem_stft.real.dtype)
            if gate_confidence is not None:
                gate_confidence = gate_confidence.to(device=stem_mag.device, dtype=stem_stft.real.dtype)
                delta_activity_gate = delta_activity_gate * gate_confidence

        owned_raw = stem_mag / stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
        peakiness = owned_raw / owned_raw.amax(dim=1, keepdim=True).clamp_min(eps)
        residual_ratio = residual_mag / mix_mag
        stem_to_mix = stem_mag / mix_mag
        phase_agreement = (
            (stem_stft.real * residual_stft.real) + (stem_stft.imag * residual_stft.imag)
        ) / (stem_mag * residual_mag + eps)
        phase_agreement = torch.clamp(phase_agreement, min=-1.0, max=1.0)

        top2 = torch.topk(owned_raw, k=min(2, owned_raw.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            ownership_margin = top2[:, :1]
        else:
            ownership_margin = top2[:, :1] - top2[:, 1:2]
        ownership_margin = ownership_margin.expand_as(owned_raw)

        features = torch.stack(
            (
                owned_raw,
                peakiness,
                residual_ratio.expand_as(owned_raw),
                stem_to_mix,
                phase_agreement,
                ownership_margin,
                torch.log1p(stem_mag),
            ),
            dim=2,
        )
        batch_size, num_stems = features.shape[:2]
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.router_type == "conv":
            logits = self.net(rearrange(features, "b n c f t -> (b n) c f t"))
            logits = rearrange(logits, "(b n) 1 f t -> b n f t", b=batch_size, n=num_stems)
        else:
            logits = self.net(features)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)

        base_weights = (owned_raw * peakiness).pow(self.base_weight_gamma)
        base_weights = base_weights / base_weights.sum(dim=1, keepdim=True).clamp_min(eps)

        stem_summary = self._build_stem_summary_features(
            owned_raw=owned_raw,
            peakiness=peakiness,
            residual_ratio=residual_ratio,
            stem_to_mix=stem_to_mix,
            phase_agreement=phase_agreement,
            ownership_margin=ownership_margin,
            stem_mag=stem_mag,
            mix_mag=mix_mag,
            residual_mag=residual_mag,
            base_weights=base_weights,
            eps=eps,
        )
        condition_outputs = self._compute_context_condition(
            base_model=base_model,
            mix_audio=mix_audio,
            stem_summary=stem_summary,
            device=device,
            dtype=logits.dtype,
        )

        blend_logit = self.blend_logit.to(device=device, dtype=logits.dtype).view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        condition_router_bias = None
        condition_delta_gain = None
        judge_score = None
        blend_delta = None
        if condition_outputs is not None:
            judge_score = condition_outputs["judge_score"]
            condition_router_bias = condition_outputs["router_bias"]
            condition_delta_gain = condition_outputs["delta_gain"]
            blend_delta = condition_outputs["blend_delta"]
            blend_logit = blend_logit + rearrange(blend_delta, "b -> b 1 1 1")

        blend = torch.sigmoid(blend_logit)
        if self.blend_floor > 0.0:
            blend = self.blend_floor + (1.0 - self.blend_floor) * blend
        delta_logits = self.delta_scale * torch.tanh(logits)
        corrected_log_weights = torch.log(base_weights.clamp_min(eps)) + blend * delta_logits
        if condition_router_bias is not None:
            corrected_log_weights = corrected_log_weights + rearrange(condition_router_bias, "b n -> b n 1 1")
        closure_prior_weights = torch.softmax(corrected_log_weights, dim=1)
        if stem_activity_mask is not None:
            corrected_log_weights = corrected_log_weights.masked_fill(
                ~stem_activity_mask.expand_as(corrected_log_weights),
                -1e4,
            )
        learned_weights = torch.softmax(corrected_log_weights, dim=1)
        if stem_activity_mask is not None:
            learned_weights = learned_weights * stem_activity_mask.to(dtype=learned_weights.dtype)
        if self.learn_residual_scale:
            residual_scale = torch.exp(
                torch.clamp(
                    self.log_residual_scale,
                    min=math.log(self.residual_scale_min),
                    max=math.log(self.residual_scale_max),
                )
            ).to(device=device, dtype=logits.dtype)
        else:
            residual_scale = self.fixed_residual_scale.to(device=device, dtype=logits.dtype)
        weights = learned_weights
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

        direct_delta_stft = torch.zeros_like(stem_stft)
        if self.delta_net is not None:
            delta_features = torch.stack(
                (
                    owned_raw,
                    peakiness,
                    residual_ratio.expand_as(owned_raw),
                    stem_to_mix,
                    phase_agreement,
                    ownership_margin,
                    torch.log1p(stem_mag),
                    stem_stft.real / mix_mag,
                    stem_stft.imag / mix_mag,
                    residual_stft.real.expand_as(owned_raw) / mix_mag,
                    residual_stft.imag.expand_as(owned_raw) / mix_mag,
                ),
                dim=2,
            )
            delta_features = torch.nan_to_num(delta_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
            delta_update = self.delta_net(rearrange(delta_features, "b n c f t -> (b n) c f t"))
            delta_update = rearrange(delta_update, "(b n) c f t -> b n c f t", b=batch_size, n=num_stems)
            delta_real = torch.tanh(delta_update[:, :, 0])
            delta_imag = torch.tanh(delta_update[:, :, 1])
            delta_norm = mix_mag.expand_as(delta_real)
            direct_delta_stft = self.delta_branch_scale * delta_norm.to(delta_real.dtype) * torch.complex(
                delta_real,
                delta_imag,
            )
            if condition_delta_gain is not None:
                direct_delta_stft = direct_delta_stft * rearrange(
                    condition_delta_gain.to(device=direct_delta_stft.device, dtype=direct_delta_stft.real.dtype),
                    "b n -> b n 1 1",
                )
            if delta_activity_gate is not None:
                direct_delta_stft = direct_delta_stft * delta_activity_gate
            elif stem_activity_mask is not None:
                direct_delta_stft = direct_delta_stft * stem_activity_mask.to(
                    device=direct_delta_stft.device,
                    dtype=direct_delta_stft.real.dtype,
                )

        refined_stft = (
            stem_stft
            + residual_scale * weights.to(stem_stft.dtype) * residual_stft
            + direct_delta_stft.to(stem_stft.dtype)
        )
        refined_audio = self._flattened_stft_to_audio(
            base_model,
            refined_stft,
            device,
            mix_audio.shape[-1],
        )
        refined_audio = torch.nan_to_num(refined_audio.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pre_closure_audio = refined_audio
        closure_weights = None
        closure_residual = None
        if self.use_exact_mix_closure:
            closure_prior = closure_prior_weights.mean(dim=(2, 3))
            refined_audio, closure_weights, closure_residual = self._apply_waveform_mix_closure(
                refined_audio,
                mix_audio,
                prior=closure_prior,
                candidate_mask=stem_activity_mask,
            )

        screen_residual_scale = refined_audio.new_tensor(0.0)
        if (
            self.router_type == "time_screening"
            and hasattr(self.net, "screen")
            and hasattr(self.net.screen, "residual_scale")
        ):
            screen_residual_scale = torch.tanh(
                self.net.screen.residual_scale
            ).to(device=refined_audio.device, dtype=refined_audio.dtype)

        judge_teacher = None
        if target_audio is not None and judge_score is not None:
            judge_teacher = self._build_judge_teacher(
                base_model=base_model,
                target_audio=target_audio,
                stem_mag=stem_mag,
                mix_mag=mix_mag,
                device=device,
                eps=eps,
            ).to(device=refined_audio.device, dtype=refined_audio.dtype)

        artifact_teacher = None
        artifact_loss_mask = None
        artifact_protect_mask = None
        artifact_base_active_mask = None
        if target_audio is not None and artifact_logits is not None:
            artifact_targets = self._build_artifact_teacher(
                base_model=base_model,
                target_audio=target_audio,
                pre_detector_stft=pre_artifact_stft,
                mix_stft=mix_stft,
                device=device,
                eps=eps,
            )
            artifact_teacher = artifact_targets["artifact_teacher"].to(
                device=artifact_logits.device,
                dtype=artifact_logits.dtype,
            )
            artifact_loss_mask = artifact_targets["artifact_loss_mask"].to(
                device=artifact_logits.device,
                dtype=artifact_logits.dtype,
            )
            artifact_protect_mask = artifact_targets["artifact_protect_mask"].to(
                device=artifact_logits.device,
                dtype=artifact_logits.dtype,
            )
            artifact_base_active_mask = artifact_targets["artifact_base_active_mask"].to(
                device=artifact_logits.device,
                dtype=artifact_logits.dtype,
            )

        aux = {
            "weights": weights,
            "base_weights": base_weights,
            "learned_weights": learned_weights,
            "delta_logits": delta_logits.detach(),
            "blend": blend.detach(),
            "residual_scale": residual_scale.detach(),
            "screen_residual_scale": screen_residual_scale.detach(),
            "residual_stft": residual_stft.detach(),
            "direct_delta_stft": direct_delta_stft.detach(),
            "reclaimed_stft": reclaimed_stft.detach(),
            "reclaimed_residual_stft": reclaimed_stft.sum(dim=1, keepdim=True).detach(),
            "artifact_logits": artifact_logits,
            "artifact_prob": None if artifact_prob is None else artifact_prob.detach(),
            "artifact_keep_gate": None if artifact_keep_gate is None else artifact_keep_gate.detach(),
            "artifact_removed_stft": artifact_removed_stft.detach(),
            "artifact_teacher": None if artifact_teacher is None else artifact_teacher.detach(),
            "artifact_loss_mask": None if artifact_loss_mask is None else artifact_loss_mask.detach(),
            "artifact_protect_mask": None if artifact_protect_mask is None else artifact_protect_mask.detach(),
            "artifact_base_active_mask": (
                None if artifact_base_active_mask is None else artifact_base_active_mask.detach()
            ),
            "silent_reclaim_gate": None if silent_reclaim_gate is None else silent_reclaim_gate.detach(),
            "judge_score": judge_score,
            "judge_teacher": None if judge_teacher is None else judge_teacher.detach(),
            "condition_router_bias": None if condition_router_bias is None else condition_router_bias.detach(),
            "condition_delta_gain": None if condition_delta_gain is None else condition_delta_gain.detach(),
            "blend_delta": None if blend_delta is None else blend_delta.detach(),
            "stem_activity_mask": None if stem_activity_mask is None else stem_activity_mask.detach(),
            "stem_reclaim_keep_mask": None if reclaim_keep_gate is None else reclaim_keep_gate.detach(),
            "stem_delta_gate": None if delta_activity_gate is None else delta_activity_gate.detach(),
            "gate_confidence": None if gate_confidence is None else gate_confidence.detach(),
            "pre_closure_audio": pre_closure_audio.detach(),
            "closure_weights": None if closure_weights is None else closure_weights.detach(),
            "closure_residual": None if closure_residual is None else closure_residual.detach(),
        }
        return refined_audio, aux
