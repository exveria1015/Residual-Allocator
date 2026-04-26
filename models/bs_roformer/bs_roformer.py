# Adapted from lucidrains/BS-RoFormer (MIT License).
# See THIRD_PARTY_NOTICES.md for the upstream copyright and license notice.

from functools import partial

import torch
from torch import nn, einsum, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.bs_roformer.attend import Attend

from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

try:
    from PoPE_pytorch import PoPE, flash_attn_with_pope
    _HAS_POPE = True
except Exception:
    PoPE = None
    flash_attn_with_pope = None
    _HAS_POPE = False

try:
    from neuralop.models import FNO
    _HAS_NEURALOP = True
except Exception:
    FNO = None
    _HAS_NEURALOP = False

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# norm

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


def tanh_norm(t, eps = 1e-8):
    norm = torch.linalg.vector_norm(t, dim = -1, keepdim = True)
    return t * (torch.tanh(norm) / norm.clamp(min = eps))


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            pope_embed=None
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.pope_embed = pope_embed
        assert not (self.rotary_embed is not None and self.pope_embed is not None), \
            "cannot have both rotary and pope embeddings"

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.pope_embed):
            assert _HAS_POPE, "PoPE requested but PoPE_pytorch is not installed"
            out = flash_attn_with_pope(
                q, k, v,
                pos_emb=self.pope_embed(q.shape[-2]),
                softmax_scale=self.scale
            )
        elif exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
            out = self.attend(q, k, v)
        else:
            out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class ZeroModule(Module):
    def forward(self, x):
        return torch.zeros_like(x)


class TimeScreeningSelector(Module):
    @beartype
    def __init__(
            self,
            *,
            dim,
            heads = 4,
            dim_head = 32,
            dropout = 0.,
            rotary_embed = None,
            norm_values = False,
            use_tanh_norm = True,
            init_window = 64.,
            init_relevance_width = 4.,
            init_scale = 0.
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.norm = RMSNorm(dim)
        self.rotary_embed = rotary_embed
        self.norm_values = norm_values
        self.use_tanh_norm = use_tanh_norm

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_inner, bias = False)
        self.to_gates = nn.Linear(dim, heads)

        init_window = max(float(init_window), 1.0001)
        init_relevance_width = max(float(init_relevance_width), 1.0001)

        self.log_window = nn.Parameter(torch.log(torch.full((heads,), init_window - 1.)))
        self.log_relevance_width = nn.Parameter(torch.log(torch.full((heads,), init_relevance_width - 1.)))
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        q = l2norm(q)
        k = l2norm(k)

        if self.norm_values:
            v = l2norm(v)

        similarity = einsum('b h i d, b h j d -> b h i j', q, k).clamp(min = -1., max = 1.)

        relevance_width = (self.log_relevance_width.exp() + 1.).to(dtype = similarity.dtype, device = similarity.device)
        relevance_width = rearrange(relevance_width, 'h -> 1 h 1 1')
        relevance = torch.clamp(1. - relevance_width * (1. - similarity), min = 0.).square()

        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device = x.device, dtype = similarity.dtype)
        offsets = (positions.view(1, -1) - positions.view(-1, 1)).abs().unsqueeze(0)

        window = (self.log_window.exp() + 1.).to(dtype = similarity.dtype, device = similarity.device)
        window = rearrange(window, 'h -> h 1 1')
        softmask = torch.where(
            offsets < window,
            0.5 * (torch.cos(torch.pi * offsets / window.clamp(min = 1.)) + 1.),
            torch.zeros_like(offsets)
        )

        relevance = relevance * softmask.unsqueeze(0)
        screened = einsum('b h i j, b h j d -> b h i d', relevance, v)

        if self.use_tanh_norm:
            screened = tanh_norm(screened)

        gates = rearrange(self.to_gates(x).sigmoid(), 'b n h -> b h n 1')
        screened = screened * gates

        out = rearrange(screened, 'b h n d -> b n (h d)')
        return self.to_out(out) * torch.tanh(self.residual_scale)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            pope_embed=None,
            flash_attn=True,
            linear_attn=False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    pope_embed=pope_embed,
                    flash=flash_attn
                )

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


def zero_init_linear_layers(module: nn.Module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if exists(child.bias):
                nn.init.zeros_(child.bias)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class ScalarBandEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            band_bin_inputs: Tuple[int, ...],
            depth,
            num_outputs_per_bin: int,
            mlp_expansion_factor=4
    ):
        super().__init__()
        assert num_outputs_per_bin > 0, "num_outputs_per_bin must be positive"
        self.band_bin_inputs = band_bin_inputs
        self.num_outputs_per_bin = int(num_outputs_per_bin)
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for band_bins in band_bin_inputs:
            self.to_freqs.append(
                MLP(
                    dim,
                    band_bins * self.num_outputs_per_bin,
                    dim_hidden=dim_hidden,
                    depth=depth
                )
            )

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp, band_bins in zip(x, self.to_freqs, self.band_bin_inputs):
            logits = mlp(band_features)
            logits = rearrange(
                logits,
                'b t (f n) -> b t f n',
                f=band_bins,
                n=self.num_outputs_per_bin
            )
            outs.append(logits)

        return torch.cat(outs, dim=2)


class ScreeningPartitionHead(Module):
    @beartype
    def __init__(
            self,
            num_stems: int,
            feature_dim: int = 5,
            token_dim: int = 16,
            heads: int = 2,
            dim_head: int = 8,
            ff_mult: int = 2,
    ):
        super().__init__()
        assert num_stems > 0, "num_stems must be positive"
        assert feature_dim > 0, "feature_dim must be positive"
        assert token_dim > 0, "token_dim must be positive"

        self.num_stems = int(num_stems)
        self.feature_dim = int(feature_dim)

        self.source_proj = nn.Linear(self.feature_dim, token_dim)
        self.bucket_proj = nn.Linear(self.feature_dim, token_dim)
        self.token_transformer = Transformer(
            dim=token_dim,
            depth=1,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=ff_mult,
            norm_output=True,
            flash_attn=False,
        )
        self.ownership_head = nn.Linear(token_dim, 1)
        self.refine_head = nn.Linear(token_dim, 1)
        zero_init_linear_layers(self.ownership_head)
        zero_init_linear_layers(self.refine_head)

    def forward(self, source_features, bucket_features):
        batch, num_stems, feature_dim, freq_bins, time_steps = source_features.shape
        assert num_stems == self.num_stems, (
            f"expected {self.num_stems} stems, got {num_stems}"
        )
        assert feature_dim == self.feature_dim, (
            f"expected feature_dim={self.feature_dim}, got {feature_dim}"
        )
        assert bucket_features.shape == (batch, feature_dim, freq_bins, time_steps), (
            "bucket_features shape mismatch"
        )

        source_tokens = rearrange(source_features, 'b n c f t -> (b f t) n c')
        bucket_token = rearrange(bucket_features, 'b c f t -> (b f t) 1 c')

        source_tokens = self.source_proj(source_tokens)
        bucket_token = self.bucket_proj(bucket_token)
        tokens = torch.cat((source_tokens, bucket_token), dim=1)
        tokens = self.token_transformer(tokens)

        ownership_logits = self.ownership_head(tokens).squeeze(-1)
        refine_logits = self.refine_head(tokens[:, :self.num_stems]).squeeze(-1)

        ownership_logits = rearrange(
            ownership_logits,
            '(b f t) n -> b n f t',
            b=batch,
            f=freq_bins,
            t=time_steps
        )
        refine_logits = rearrange(
            refine_logits,
            '(b f t) n -> b n f t',
            b=batch,
            f=freq_bins,
            t=time_steps
        )
        return ownership_logits[:, :self.num_stems], ownership_logits[:, self.num_stems:self.num_stems + 1], refine_logits


class TinyOwnedCalibrator(Module):
    @beartype
    def __init__(
            self,
            in_channels: int = 5,
            hidden_channels: int = 8,
            kernel_size: int = 5,
    ):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, (
            "owned calibrator kernel_size must be a positive odd integer"
        )
        padding = (kernel_size // 2, 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(kernel_size, 1), padding=padding),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        final_conv = self.net[-1]
        # Keep the block near identity, but strong enough that the learned
        # branch can escape the deterministic fallback during early training.
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        return self.net(x)


class OwnedScreeningCalibrator(Module):
    @beartype
    def __init__(
            self,
            in_channels: int = 5,
            hidden_channels: int = 8,
            kernel_size: int = 5,
    ):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, (
            "owned screening kernel_size must be a positive odd integer"
        )
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.to_g = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.to_out = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        init_window = max(float(self.pad + 1), 1.0001)
        init_relevance_width = 4.0
        self.log_window = nn.Parameter(torch.log(torch.tensor(init_window - 1.0, dtype=torch.float32)))
        self.log_relevance_width = nn.Parameter(
            torch.log(torch.tensor(init_relevance_width - 1.0, dtype=torch.float32))
        )

        # Keep the initial perturbation small, but not so tiny that the
        # screening path gets dominated by deterministic fallback forever.
        nn.init.normal_(self.to_out.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x):
        eps = 1e-8
        b, _, f, t = x.shape

        q = F.normalize(self.to_q(x), dim=1, eps=eps)
        k = F.normalize(self.to_k(x), dim=1, eps=eps)
        v = F.normalize(self.to_v(x), dim=1, eps=eps)
        g = torch.tanh(F.silu(self.to_g(x)))

        q = rearrange(q, 'b d f t -> (b t) f d')
        k = rearrange(k, 'b d f t -> (b t) f d')
        v = rearrange(v, 'b d f t -> (b t) f d')
        g = rearrange(g, 'b d f t -> (b t) f d')

        k_padded = F.pad(k.permute(0, 2, 1), (self.pad, self.pad), mode='replicate').permute(0, 2, 1)
        v_padded = F.pad(v.permute(0, 2, 1), (self.pad, self.pad), mode='replicate').permute(0, 2, 1)
        k_local = k_padded.unfold(1, self.kernel_size, 1).permute(0, 1, 3, 2)
        v_local = v_padded.unfold(1, self.kernel_size, 1).permute(0, 1, 3, 2)

        similarity = (q.unsqueeze(2) * k_local).sum(dim=-1).clamp(min=-1.0, max=1.0)
        relevance_width = torch.nan_to_num(self.log_relevance_width, nan=0.0).exp() + 1.0
        relevance_width = relevance_width.clamp(min=1.0, max=16.0).to(
            device=similarity.device, dtype=similarity.dtype
        )
        relevance = torch.clamp(1.0 - relevance_width * (1.0 - similarity), min=0.0).square()

        offsets = torch.arange(-self.pad, self.pad + 1, device=similarity.device, dtype=similarity.dtype).abs()
        window = torch.nan_to_num(self.log_window, nan=0.0).exp() + 1.0
        window = window.clamp(min=1.0, max=float(self.kernel_size)).to(
            device=similarity.device, dtype=similarity.dtype
        )
        softmask = torch.where(
            offsets < window,
            0.5 * (torch.cos(torch.pi * offsets / window.clamp(min=1.0)) + 1.0),
            torch.zeros_like(offsets)
        )

        relevance = relevance * softmask.view(1, 1, self.kernel_size)
        screened = (relevance.unsqueeze(-1) * v_local).sum(dim=2)
        screened = tanh_norm(screened) * g
        screened = rearrange(screened, '(b t) f d -> b d f t', b=b, t=t)
        return self.to_out(screened)


class OwnedBandDeltaCalibrator(Module):
    @beartype
    def __init__(
            self,
            num_stems: int,
            num_bins: int,
    ):
        super().__init__()
        assert num_stems > 0, "owned band delta calibrator requires a positive number of stems"
        assert num_bins > 0, "owned band delta calibrator requires a positive number of bins"
        self.num_stems = int(num_stems)
        self.num_bins = int(num_bins)
        self.delta = nn.Parameter(torch.zeros(1, self.num_stems, self.num_bins, 1, dtype=torch.float32))

    def forward(self, x):
        batch_stems, _, freq_bins, time_steps = x.shape
        assert freq_bins == self.num_bins, (
            f"owned band delta calibrator expected {self.num_bins} bins, got {freq_bins}"
        )
        assert batch_stems % self.num_stems == 0, (
            "owned band delta calibrator received a batch that is not divisible by num_stems"
        )
        batch_size = batch_stems // self.num_stems
        delta = self.delta.expand(batch_size, -1, -1, time_steps)
        return rearrange(delta, 'b n f t -> (b n) 1 f t')


class OwnedTemporalDeltaCalibrator(Module):
    @beartype
    def __init__(
            self,
            in_channels: int = 5,
            hidden_channels: int = 8,
            kernel_size: int = 3,
    ):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, (
            "owned temporal delta calibrator kernel_size must be a positive odd integer"
        )
        padding = (0, kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, kernel_size), padding=padding),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        final_conv = self.net[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        return self.net(x)


class FNOMaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            fno_n_modes=64,
            fno_n_layers=3,
            fno_separable=True
    ):
        super().__init__()
        assert _HAS_NEURALOP, "mask_estimator_type='fno' requires neuraloperator / neuralop to be installed"

        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])

        for dim_in in dim_inputs:
            self.to_freqs.append(
                nn.Sequential(
                    FNO(
                        n_modes=(fno_n_modes,),
                        hidden_channels=dim,
                        in_channels=dim,
                        out_channels=dim_in * 2,
                        n_layers=fno_n_layers,
                        separable=fno_separable
                    ),
                    nn.GLU(dim=-2)
                )
            )

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, fno in zip(x, self.to_freqs):
            band_features = rearrange(band_features, 'b t c -> b c t')

            if band_features.is_cuda:
                with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
                    freq_out = fno(band_features.float()).float()
            else:
                freq_out = fno(band_features.float()).float()

            freq_out = rearrange(freq_out, 'b c t -> b t c')
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            zero_dc = True,
            mask_estimator_depth=2,
            mask_estimator_type: str = 'mlp',
            output_head_type: str = 'mask',
            partition_head_depth: Optional[int] = None,
            partition_head_mlp_expansion_factor: Optional[int] = None,
            partition_bucket_init_bias: float = -8.0,
            partition_screening_token_dim: int = 16,
            partition_screening_heads: int = 2,
            partition_screening_dim_head: int = 8,
            partition_refine_supervision_loss_weight: float = 0.,
            fno_n_modes: int = 64,
            fno_n_layers: int = 3,
            fno_separable: bool = True,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            use_pope: bool = False,
            use_time_screening: bool = False,
            time_screening_heads: int = 4,
            time_screening_dim_head: int = 32,
            time_screening_dropout: float = 0.,
            time_screening_norm_values: bool = False,
            time_screening_tanh_norm: bool = True,
            time_screening_init_window: float = 64.,
            time_screening_init_relevance_width: float = 4.,
            time_screening_init_scale: float = 0.,
            time_screening_layer_start: Optional[int] = None,
            use_final_time_screening: bool = False,
            final_time_screening_heads: Optional[int] = None,
            final_time_screening_dim_head: Optional[int] = None,
            final_time_screening_dropout: Optional[float] = None,
            final_time_screening_norm_values: Optional[bool] = None,
            final_time_screening_tanh_norm: Optional[bool] = None,
            final_time_screening_init_window: Optional[float] = None,
            final_time_screening_init_relevance_width: Optional[float] = None,
            final_time_screening_init_scale: Optional[float] = None,
            stem_loss_weights: Optional[Tuple[float, ...]] = None,
            mix_consistency_loss_weight: float = 0.,
            vocal_complement_loss_weight: float = 0.,
            stem_complement_loss_weights: Optional[Tuple[float, ...]] = None,
            use_residual_add_back_router: bool = False,
            residual_add_back_init_scale: float = 0.,
            residual_router_stem_priors: Optional[Tuple[float, ...]] = None,
            use_pre_final_aux_head: bool = False,
            pre_final_aux_head_loss_weight: float = 0.,
            use_post_mask_mlp_correction: bool = False,
            post_mask_correction_depth: int = 2,
            post_mask_correction_mlp_expansion_factor: Optional[int] = None,
            post_mask_correction_init_scale: float = 0.,
            post_mask_delta_loss_weight: float = 0.,
            post_mask_delta_consistency_loss_weight: float = 0.,
            use_dual_output_heads: bool = False,
            dual_clean_head_depth: int = 2,
            dual_clean_head_mlp_expansion_factor: Optional[int] = None,
            dual_gate_head_depth: int = 2,
            dual_gate_head_mlp_expansion_factor: Optional[int] = None,
            dual_gate_init_bias: float = -4.,
            dual_full_loss_weight: float = 0.25,
            dual_clean_owned_loss_weight: float = 0.05,
            dual_clean_forbidden_loss_weight: float = 0.05,
            dual_clean_owned_threshold: float = 0.6,
            dual_clean_forbidden_threshold: float = 0.05,
            use_clean_band_screening: bool = False,
            clean_band_screening_heads: int = 4,
            clean_band_screening_dim_head: int = 32,
            clean_band_screening_dropout: float = 0.,
            clean_band_screening_norm_values: bool = False,
            clean_band_screening_tanh_norm: bool = True,
            clean_band_screening_init_window: float = 8.,
            clean_band_screening_init_relevance_width: float = 4.,
            clean_band_screening_init_scale: float = 0.,
            use_residual_only_refiner: bool = False,
            residual_refiner_depth: int = 2,
            residual_refiner_mlp_expansion_factor: Optional[int] = None,
            residual_refiner_init_scale: float = 0.,
            residual_refiner_stem_priors: Optional[Tuple[float, ...]] = None,
            use_owned_calibrator: bool = False,
            owned_calibrator_type: str = 'conv',
            owned_calibrator_hidden_channels: int = 8,
            owned_calibrator_kernel_size: int = 5,
            owned_calibrator_threshold: float = 0.6,
            owned_calibrator_train_threshold: Optional[float] = None,
            owned_calibrator_soft_threshold_width: float = 0.05,
            owned_calibrator_gamma: float = 2.0,
            owned_calibrator_residual_scale: float = 1.0,
            owned_calibrator_train_fallback_mix: float = 0.0,
            owned_calibrator_delta_scale: float = 4.0,
            owned_calibrator_init_scale: float = 0.,
            use_exact_mix_closure: bool = False,
            exact_mix_closure_topk: int = 0,
            use_delta_trunk: bool = False,
            delta_trunk_depth: int = 2,
            delta_trunk_time_transformer_depth: Optional[int] = None,
            delta_trunk_freq_transformer_depth: Optional[int] = None,
            delta_trunk_linear_transformer_depth: Optional[int] = None,
            delta_trunk_init_scale: float = 0.,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.model_dim = int(dim)
        self.refiner_condition_dim = int(dim) * 2
        self.dim_freqs_in = dim_freqs_in
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.mask_estimator_type = mask_estimator_type.lower()
        self.mix_consistency_loss_weight = float(mix_consistency_loss_weight)
        self.vocal_complement_loss_weight = float(vocal_complement_loss_weight)
        self.use_residual_add_back_router = bool(use_residual_add_back_router)
        self.pre_final_aux_head_loss_weight = float(pre_final_aux_head_loss_weight)
        self.use_pre_final_aux_head = bool(use_pre_final_aux_head)
        self.use_post_mask_mlp_correction = bool(use_post_mask_mlp_correction)
        self.post_mask_delta_loss_weight = float(post_mask_delta_loss_weight)
        self.post_mask_delta_consistency_loss_weight = float(post_mask_delta_consistency_loss_weight)
        self.use_dual_output_heads = bool(use_dual_output_heads)
        self.dual_full_loss_weight = float(dual_full_loss_weight)
        self.dual_clean_owned_loss_weight = float(dual_clean_owned_loss_weight)
        self.dual_clean_forbidden_loss_weight = float(dual_clean_forbidden_loss_weight)
        self.dual_clean_owned_threshold = float(dual_clean_owned_threshold)
        self.dual_clean_forbidden_threshold = float(dual_clean_forbidden_threshold)
        self.use_clean_band_screening = bool(use_clean_band_screening)
        self.use_residual_only_refiner = bool(use_residual_only_refiner)
        self.use_owned_calibrator = bool(use_owned_calibrator)
        self.output_head_type = str(output_head_type).lower()
        self.partition_refine_supervision_loss_weight = float(partition_refine_supervision_loss_weight)
        self.owned_calibrator_type = str(owned_calibrator_type).lower()
        self.owned_calibrator_threshold = float(owned_calibrator_threshold)
        self.owned_calibrator_train_threshold = (
            None if owned_calibrator_train_threshold is None
            else float(owned_calibrator_train_threshold)
        )
        self.owned_calibrator_soft_threshold_width = float(owned_calibrator_soft_threshold_width)
        self.owned_calibrator_gamma = float(owned_calibrator_gamma)
        self.owned_calibrator_residual_scale = float(owned_calibrator_residual_scale)
        self.owned_calibrator_train_fallback_mix = float(owned_calibrator_train_fallback_mix)
        self.owned_calibrator_delta_scale = float(owned_calibrator_delta_scale)
        self.use_exact_mix_closure = bool(use_exact_mix_closure)
        self.exact_mix_closure_topk = int(exact_mix_closure_topk)
        self.use_delta_trunk = bool(use_delta_trunk)

        assert not (
            self.use_post_mask_mlp_correction and self.use_dual_output_heads
        ), "use_post_mask_mlp_correction and use_dual_output_heads are mutually exclusive"
        assert not (
            self.use_clean_band_screening and not self.use_dual_output_heads
        ), "use_clean_band_screening requires use_dual_output_heads=True"
        assert not (
            self.use_residual_add_back_router and self.use_residual_only_refiner
        ), "use_residual_add_back_router and use_residual_only_refiner are mutually exclusive"
        assert 0.0 <= self.dual_clean_forbidden_threshold < self.dual_clean_owned_threshold <= 1.0, (
            "dual_clean thresholds must satisfy 0 <= forbidden < owned <= 1"
        )
        assert 0.0 <= self.owned_calibrator_threshold <= 1.0, (
            "owned_calibrator_threshold must be between 0 and 1"
        )
        assert (
            self.owned_calibrator_train_threshold is None
            or 0.0 <= self.owned_calibrator_train_threshold <= 1.0
        ), "owned_calibrator_train_threshold must be between 0 and 1"
        assert self.owned_calibrator_soft_threshold_width > 0.0, (
            "owned_calibrator_soft_threshold_width must be positive"
        )
        assert self.owned_calibrator_train_fallback_mix >= 0.0, (
            "owned_calibrator_train_fallback_mix must be non-negative"
        )
        assert self.exact_mix_closure_topk >= 0, (
            "exact_mix_closure_topk must be non-negative"
        )
        assert self.owned_calibrator_type in {'conv', 'screening', 'band_delta', 'temporal_delta'}, (
            "owned_calibrator_type must be one of: 'conv', 'screening', 'band_delta', 'temporal_delta'"
        )
        assert self.output_head_type in {'mask', 'partition', 'partition_screening'}, (
            "output_head_type must be one of: 'mask', 'partition', 'partition_screening'"
        )
        assert self.partition_refine_supervision_loss_weight >= 0.0, (
            "partition_refine_supervision_loss_weight must be non-negative"
        )
        if self.output_head_type in {'partition', 'partition_screening'}:
            assert not self.use_dual_output_heads, (
                "partition output heads do not support dual output heads in the minimal implementation"
            )
            assert not self.use_post_mask_mlp_correction, (
                "partition output heads do not support post-mask correction in the minimal implementation"
            )
            assert not self.use_residual_add_back_router, (
                "partition output heads do not support residual add-back router in the minimal implementation"
            )
            assert not self.use_residual_only_refiner, (
                "partition output heads do not support residual-only refiner in the minimal implementation"
            )
            assert not self.use_owned_calibrator, (
                "partition output heads are mutually exclusive with use_owned_calibrator in the minimal implementation"
            )

        assert self.mask_estimator_type in {'mlp', 'fno'}, "mask_estimator_type must be one of: 'mlp', 'fno'"
        if self.mask_estimator_type == 'fno':
            assert _HAS_NEURALOP, "mask_estimator_type='fno' requires neuraloperator / neuralop to be installed"

        if stem_loss_weights is None:
            stem_loss_weights = (1.,) * num_stems
        else:
            stem_loss_weights = tuple(float(weight) for weight in stem_loss_weights)
            assert len(stem_loss_weights) == num_stems, (
                f"stem_loss_weights must have length {num_stems}, got {len(stem_loss_weights)}"
            )
        self.register_buffer(
            'stem_loss_weights',
            torch.tensor(stem_loss_weights, dtype=torch.float32),
            persistent=False
        )

        if stem_complement_loss_weights is None:
            stem_complement_loss_weights = (0.,) * num_stems
        else:
            stem_complement_loss_weights = tuple(float(weight) for weight in stem_complement_loss_weights)
            assert len(stem_complement_loss_weights) == num_stems, (
                f"stem_complement_loss_weights must have length {num_stems}, "
                f"got {len(stem_complement_loss_weights)}"
            )

        # Backward-compatible shorthand for the previous vocal-only residual loss.
        if self.vocal_complement_loss_weight > 0. and num_stems > 0:
            stem_complement_loss_weights = list(stem_complement_loss_weights)
            stem_complement_loss_weights[-1] += self.vocal_complement_loss_weight
            stem_complement_loss_weights = tuple(stem_complement_loss_weights)

        self.register_buffer(
            'stem_complement_loss_weights',
            torch.tensor(stem_complement_loss_weights, dtype=torch.float32),
            persistent=False
        )

        if residual_router_stem_priors is None:
            residual_router_stem_priors = (1.,) * num_stems
        else:
            residual_router_stem_priors = tuple(float(weight) for weight in residual_router_stem_priors)
            assert len(residual_router_stem_priors) == num_stems, (
                f"residual_router_stem_priors must have length {num_stems}, "
                f"got {len(residual_router_stem_priors)}"
            )
        self.register_buffer(
            'residual_router_stem_priors',
            torch.tensor(residual_router_stem_priors, dtype=torch.float32),
            persistent=False
        )

        if residual_refiner_stem_priors is None:
            residual_refiner_stem_priors = (1.,) * num_stems
        else:
            residual_refiner_stem_priors = tuple(float(weight) for weight in residual_refiner_stem_priors)
            assert len(residual_refiner_stem_priors) == num_stems, (
                f"residual_refiner_stem_priors must have length {num_stems}, "
                f"got {len(residual_refiner_stem_priors)}"
            )
        self.register_buffer(
            'residual_refiner_stem_priors',
            torch.tensor(residual_refiner_stem_priors, dtype=torch.float32),
            persistent=False
        )

        self.layers = ModuleList([])
        self.time_screening_selectors = ModuleList([])
        self.final_time_screening_selector = ZeroModule()

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
        )

        if use_pope:
            assert _HAS_POPE, "PoPE requested but PoPE_pytorch is not installed"
            time_pope_embed = PoPE(dim=dim_head, heads=heads)
            freq_pope_embed = PoPE(dim=dim_head, heads=heads)
            time_rotary_embed = None
            freq_rotary_embed = None
            time_screening_rotary_embed = None
        else:
            time_rotary_embed = RotaryEmbedding(dim = dim_head)
            freq_rotary_embed = RotaryEmbedding(dim = dim_head)
            time_screening_rotary_embed = RotaryEmbedding(dim = time_screening_dim_head)
            time_pope_embed = freq_pope_embed = None

        if exists(time_screening_layer_start) and time_screening_layer_start < 0:
            time_screening_layer_start = max(depth + time_screening_layer_start, 0)

        for layer_index in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    pope_embed=time_pope_embed,
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    pope_embed=freq_pope_embed,
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

            should_use_time_screening = use_time_screening and (
                not exists(time_screening_layer_start) or layer_index >= time_screening_layer_start
            )

            if should_use_time_screening:
                self.time_screening_selectors.append(
                    TimeScreeningSelector(
                        dim = dim,
                        heads = time_screening_heads,
                        dim_head = time_screening_dim_head,
                        dropout = time_screening_dropout,
                        rotary_embed = time_screening_rotary_embed,
                        norm_values = time_screening_norm_values,
                        use_tanh_norm = time_screening_tanh_norm,
                        init_window = time_screening_init_window,
                        init_relevance_width = time_screening_init_relevance_width,
                        init_scale = time_screening_init_scale
                    )
                )
            else:
                self.time_screening_selectors.append(ZeroModule())

        if use_final_time_screening:
            final_time_screening_heads = default(final_time_screening_heads, time_screening_heads)
            final_time_screening_dim_head = default(final_time_screening_dim_head, time_screening_dim_head)
            final_time_screening_dropout = default(final_time_screening_dropout, time_screening_dropout)
            final_time_screening_norm_values = default(final_time_screening_norm_values, time_screening_norm_values)
            final_time_screening_tanh_norm = default(final_time_screening_tanh_norm, time_screening_tanh_norm)
            final_time_screening_init_window = default(final_time_screening_init_window, time_screening_init_window)
            final_time_screening_init_relevance_width = default(
                final_time_screening_init_relevance_width,
                time_screening_init_relevance_width
            )
            final_time_screening_init_scale = default(final_time_screening_init_scale, time_screening_init_scale)
            final_time_screening_rotary_embed = RotaryEmbedding(dim = final_time_screening_dim_head)
            if use_pope:
                final_time_screening_rotary_embed = None

            self.final_time_screening_selector = TimeScreeningSelector(
                dim = dim,
                heads = final_time_screening_heads,
                dim_head = final_time_screening_dim_head,
                dropout = final_time_screening_dropout,
                rotary_embed = final_time_screening_rotary_embed,
                norm_values = final_time_screening_norm_values,
                use_tanh_norm = final_time_screening_tanh_norm,
                init_window = final_time_screening_init_window,
                init_relevance_width = final_time_screening_init_relevance_width,
                init_scale = final_time_screening_init_scale
            )

        self.delta_trunk_layers = ModuleList([])
        self.delta_trunk_scale = None
        if self.use_delta_trunk:
            delta_trunk_time_transformer_depth = default(
                delta_trunk_time_transformer_depth,
                time_transformer_depth
            )
            delta_trunk_freq_transformer_depth = default(
                delta_trunk_freq_transformer_depth,
                freq_transformer_depth
            )
            delta_trunk_linear_transformer_depth = default(
                delta_trunk_linear_transformer_depth,
                linear_transformer_depth
            )

            for _ in range(delta_trunk_depth):
                tran_modules = []
                if delta_trunk_linear_transformer_depth > 0:
                    tran_modules.append(
                        Transformer(
                            depth=delta_trunk_linear_transformer_depth,
                            linear_attn=True,
                            **transformer_kwargs
                        )
                    )
                tran_modules.append(
                    Transformer(
                        depth=delta_trunk_time_transformer_depth,
                        rotary_embed=time_rotary_embed,
                        pope_embed=time_pope_embed,
                        **transformer_kwargs
                    )
                )
                tran_modules.append(
                    Transformer(
                        depth=delta_trunk_freq_transformer_depth,
                        rotary_embed=freq_rotary_embed,
                        pope_embed=freq_pope_embed,
                        **transformer_kwargs
                    )
                )
                self.delta_trunk_layers.append(nn.ModuleList(tran_modules))

            self.delta_trunk_scale = nn.Parameter(
                torch.tensor(float(delta_trunk_init_scale), dtype=torch.float32)
            )

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        freqs_per_bands_with_audio_channels = tuple(f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        if self.mask_estimator_type == 'fno':
            mask_estimator_cls = FNOMaskEstimator
            mask_estimator_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                fno_n_modes=fno_n_modes,
                fno_n_layers=fno_n_layers,
                fno_separable=fno_separable,
            )
        else:
            mask_estimator_cls = MaskEstimator
            mask_estimator_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

        for _ in range(num_stems):
            mask_estimator = mask_estimator_cls(**mask_estimator_kwargs)

            self.mask_estimators.append(mask_estimator)

        self.partition_source_delta_head = None
        self.partition_bucket_head = None
        self.partition_refine_delta_head = None
        self.partition_screening_head = None
        self.partition_bucket_bias = None
        if self.output_head_type == 'partition':
            partition_head_depth = default(partition_head_depth, mask_estimator_depth)
            partition_head_mlp_expansion_factor = default(
                partition_head_mlp_expansion_factor,
                mlp_expansion_factor
            )
            partition_head_kwargs = dict(
                dim=dim,
                band_bin_inputs=freqs_per_bands_with_audio_channels,
                depth=partition_head_depth,
                mlp_expansion_factor=partition_head_mlp_expansion_factor,
            )
            self.partition_source_delta_head = ScalarBandEstimator(
                num_outputs_per_bin=num_stems,
                **partition_head_kwargs,
            )
            self.partition_bucket_head = ScalarBandEstimator(
                num_outputs_per_bin=1,
                **partition_head_kwargs,
            )
            self.partition_refine_delta_head = ScalarBandEstimator(
                num_outputs_per_bin=num_stems,
                **partition_head_kwargs,
            )
            zero_init_linear_layers(self.partition_source_delta_head)
            zero_init_linear_layers(self.partition_bucket_head)
            zero_init_linear_layers(self.partition_refine_delta_head)
            self.partition_bucket_bias = nn.Parameter(
                torch.tensor(float(partition_bucket_init_bias), dtype=torch.float32)
            )
        elif self.output_head_type == 'partition_screening':
            self.partition_screening_head = ScreeningPartitionHead(
                num_stems=num_stems,
                feature_dim=5,
                token_dim=partition_screening_token_dim,
                heads=partition_screening_heads,
                dim_head=partition_screening_dim_head,
            )
            self.partition_bucket_bias = nn.Parameter(
                torch.tensor(float(partition_bucket_init_bias), dtype=torch.float32)
            )

        self.pre_final_aux_mask_estimator = None
        if self.use_pre_final_aux_head:
            self.pre_final_aux_mask_estimator = mask_estimator_cls(**mask_estimator_kwargs)

        self.clean_mask_estimators = nn.ModuleList([])
        self.dual_gate_estimators = nn.ModuleList([])
        self.dual_gate_bias = None
        if self.use_dual_output_heads:
            dual_clean_head_mlp_expansion_factor = default(
                dual_clean_head_mlp_expansion_factor,
                mlp_expansion_factor
            )
            dual_gate_head_mlp_expansion_factor = default(
                dual_gate_head_mlp_expansion_factor,
                dual_clean_head_mlp_expansion_factor
            )

            clean_head_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=dual_clean_head_depth,
                mlp_expansion_factor=dual_clean_head_mlp_expansion_factor,
            )
            gate_head_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=dual_gate_head_depth,
                mlp_expansion_factor=dual_gate_head_mlp_expansion_factor,
            )

            for _ in range(num_stems):
                self.clean_mask_estimators.append(MaskEstimator(**clean_head_kwargs))
                gate_estimator = MaskEstimator(**gate_head_kwargs)
                zero_init_linear_layers(gate_estimator)
                self.dual_gate_estimators.append(gate_estimator)

            self.dual_gate_bias = nn.Parameter(
                torch.full((num_stems,), float(dual_gate_init_bias), dtype=torch.float32)
            )

        self.clean_band_screening_selectors = nn.ModuleList([])
        if self.use_clean_band_screening:
            clean_band_screening_rotary_embed = RotaryEmbedding(dim = clean_band_screening_dim_head)
            if use_pope:
                clean_band_screening_rotary_embed = None

            for _ in range(num_stems):
                self.clean_band_screening_selectors.append(
                    TimeScreeningSelector(
                        dim = dim,
                        heads = clean_band_screening_heads,
                        dim_head = clean_band_screening_dim_head,
                        dropout = clean_band_screening_dropout,
                        rotary_embed = clean_band_screening_rotary_embed,
                        norm_values = clean_band_screening_norm_values,
                        use_tanh_norm = clean_band_screening_tanh_norm,
                        init_window = clean_band_screening_init_window,
                        init_relevance_width = clean_band_screening_init_relevance_width,
                        init_scale = clean_band_screening_init_scale
                    )
                )

        self.post_mask_correction_estimators = nn.ModuleList([])
        self.post_mask_correction_scale = None
        if self.use_post_mask_mlp_correction:
            post_mask_correction_mlp_expansion_factor = default(
                post_mask_correction_mlp_expansion_factor,
                mlp_expansion_factor
            )
            post_mask_correction_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=post_mask_correction_depth,
                mlp_expansion_factor=post_mask_correction_mlp_expansion_factor,
            )
            for _ in range(num_stems):
                self.post_mask_correction_estimators.append(
                    MaskEstimator(**post_mask_correction_kwargs)
                )
            self.post_mask_correction_scale = nn.Parameter(
                torch.full((num_stems,), float(post_mask_correction_init_scale), dtype=torch.float32)
            )

        self.residual_router_estimators = nn.ModuleList([])
        self.residual_router_scale = None
        if self.use_residual_add_back_router:
            for _ in range(num_stems):
                self.residual_router_estimators.append(mask_estimator_cls(**mask_estimator_kwargs))
            self.residual_router_scale = nn.Parameter(
                torch.tensor(float(residual_add_back_init_scale), dtype=torch.float32)
            )

        self.residual_refiner_estimators = nn.ModuleList([])
        self.residual_refiner_scale = None
        if self.use_residual_only_refiner:
            residual_refiner_mlp_expansion_factor = default(
                residual_refiner_mlp_expansion_factor,
                mlp_expansion_factor
            )
            residual_refiner_kwargs = dict(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=residual_refiner_depth,
                mlp_expansion_factor=residual_refiner_mlp_expansion_factor,
            )
            for _ in range(num_stems):
                self.residual_refiner_estimators.append(
                    MaskEstimator(**residual_refiner_kwargs)
                )
            self.residual_refiner_scale = nn.Parameter(
                torch.tensor(float(residual_refiner_init_scale), dtype=torch.float32)
            )

        self.owned_calibrator = None
        self.owned_calibrator_scale = None
        if self.use_owned_calibrator:
            if self.owned_calibrator_type == 'screening':
                self.owned_calibrator = OwnedScreeningCalibrator(
                    in_channels=5,
                    hidden_channels=owned_calibrator_hidden_channels,
                    kernel_size=owned_calibrator_kernel_size,
                )
            elif self.owned_calibrator_type == 'band_delta':
                self.owned_calibrator = OwnedBandDeltaCalibrator(
                    num_stems=self.num_stems,
                    num_bins=int(self.dim_freqs_in * self.audio_channels),
                )
            elif self.owned_calibrator_type == 'temporal_delta':
                self.owned_calibrator = OwnedTemporalDeltaCalibrator(
                    in_channels=5,
                    hidden_channels=owned_calibrator_hidden_channels,
                    kernel_size=owned_calibrator_kernel_size,
                )
            else:
                self.owned_calibrator = TinyOwnedCalibrator(
                    in_channels=5,
                    hidden_channels=owned_calibrator_hidden_channels,
                    kernel_size=owned_calibrator_kernel_size,
                )
            self.owned_calibrator_scale = nn.Parameter(
                torch.tensor(float(owned_calibrator_init_scale), dtype=torch.float32)
            )

        # whether to zero out dc

        self.zero_dc = zero_dc

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def _audio_to_flattened_stft(self, audio, device):
        x_is_mps = device.type == "mps"
        stft_window = self.stft_window_fn(device=device)
        audio = audio.to(device=device, dtype=torch.float32)

        if audio.ndim == 3:
            batch_size, num_channels = audio.shape[:2]
            packed = rearrange(audio, 'b s t -> (b s) t')
            num_stems = 1
        elif audio.ndim == 4:
            batch_size, num_stems, num_channels = audio.shape[:3]
            packed = rearrange(audio, 'b n s t -> (b n s) t')
        else:
            raise ValueError(f"Unsupported audio rank for owned calibrator: {audio.shape}")

        try:
            stft = torch.stft(
                packed,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=True
            )
        except Exception:
            stft = torch.stft(
                packed.cpu() if x_is_mps else packed,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)

        if num_stems == 1:
            return rearrange(
                stft,
                '(b s) f t -> b 1 (f s) t',
                b=batch_size,
                s=num_channels
            )

        return rearrange(
            stft,
            '(b n s) f t -> b n (f s) t',
            b=batch_size,
            n=num_stems,
            s=num_channels
        )

    def _flattened_stft_to_audio(self, masked_stft, device, audio_length):
        x_is_mps = device.type == "mps"
        batch_size, num_stems = masked_stft.shape[:2]
        masked_stft = rearrange(masked_stft, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        if self.zero_dc:
            masked_stft = masked_stft.index_fill(1, tensor(0, device=device), 0.)

        stft_window = self.stft_window_fn(device=device)
        try:
            recon = torch.istft(
                masked_stft,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=audio_length
            )
        except Exception:
            recon = torch.istft(
                masked_stft.cpu() if x_is_mps else masked_stft,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=audio_length
            ).to(device)

        return rearrange(
            recon,
            '(b n s) t -> b n s t',
            b=batch_size,
            n=num_stems,
            s=self.audio_channels
        )

    def _encode_refiner_condition_tokens(self, raw_audio, device):
        x_is_mps = device.type == "mps"

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), (
            'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2).'
            ' also need to be False if mono (channel dimension of 1)'
        )

        audio_packed, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(
                audio_packed,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                audio_packed.cpu() if x_is_mps else audio_packed,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)

        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            time_screening_selector = self.time_screening_selectors[i]
            if self.use_torch_checkpoint and not isinstance(time_screening_selector, ZeroModule):
                x = x + checkpoint(time_screening_selector, x, use_reentrant=False)
            else:
                x = x + time_screening_selector(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        if len(self.delta_trunk_layers) > 0:
            delta_input = x
            delta_x = x

            for transformer_block in self.delta_trunk_layers:

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    delta_x, ft_ps = pack([delta_x], 'b * d')
                    if self.use_torch_checkpoint:
                        delta_x = checkpoint(linear_transformer, delta_x, use_reentrant=False)
                    else:
                        delta_x = linear_transformer(delta_x)
                    delta_x, = unpack(delta_x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                delta_x = rearrange(delta_x, 'b t f d -> b f t d')
                delta_x, ps = pack([delta_x], '* t d')

                if self.use_torch_checkpoint:
                    delta_x = checkpoint(time_transformer, delta_x, use_reentrant=False)
                else:
                    delta_x = time_transformer(delta_x)

                delta_x, = unpack(delta_x, ps, '* t d')
                delta_x = rearrange(delta_x, 'b f t d -> b t f d')
                delta_x, ps = pack([delta_x], '* f d')

                if self.use_torch_checkpoint:
                    delta_x = checkpoint(freq_transformer, delta_x, use_reentrant=False)
                else:
                    delta_x = freq_transformer(delta_x)

                delta_x, = unpack(delta_x, ps, '* f d')

            delta_scale = self.delta_trunk_scale.to(device=device, dtype=delta_x.dtype)
            x = delta_input + (delta_x - delta_input) * delta_scale

        if not isinstance(self.final_time_screening_selector, ZeroModule):
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = x + checkpoint(self.final_time_screening_selector, x, use_reentrant=False)
            else:
                x = x + self.final_time_screening_selector(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')

        x_pre_final_screen = x
        x = self.final_norm(x)
        return x_pre_final_screen, x

    def extract_refiner_condition_vector(self, raw_audio, device=None):
        if device is None:
            device = raw_audio.device
        x_pre_final_screen, x = self._encode_refiner_condition_tokens(raw_audio, device)
        pooled_pre = x_pre_final_screen.mean(dim=(1, 2))
        pooled_final = x.mean(dim=(1, 2))
        return torch.cat((pooled_pre, pooled_final), dim=-1)

    def extract_refiner_condition_tokens(
        self,
        raw_audio,
        device=None,
        *,
        num_frames=16,
        num_bands=8,
    ):
        if device is None:
            device = raw_audio.device
        x_pre_final_screen, x = self._encode_refiner_condition_tokens(raw_audio, device)
        tokens = torch.cat((x_pre_final_screen, x), dim=-1)
        if num_frames is not None and num_bands is not None:
            num_frames = max(1, int(num_frames))
            num_bands = max(1, int(num_bands))
            tokens = rearrange(tokens, 'b t f d -> b d t f')
            tokens = F.adaptive_avg_pool2d(tokens, output_size=(num_frames, num_bands))
            tokens = rearrange(tokens, 'b d t f -> b (t f) d')
        else:
            tokens = rearrange(tokens, 'b t f d -> b (t f) d')
        return tokens

    def _normalize_owned_weights(self, weights, fallback, eps=1e-8):
        weights_sum = weights.sum(dim=1, keepdim=True)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(eps)
        normalized = weights / weights_sum.clamp_min(eps)
        return torch.where(weights_sum > eps, normalized, fallback)

    def _apply_waveform_mix_closure(
        self,
        recon_audio,
        mix_audio,
        eps=1e-8,
        prior: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
    ):
        mix_audio = mix_audio[..., :recon_audio.shape[-1]]
        residual = mix_audio.unsqueeze(1) - recon_audio.sum(dim=1, keepdim=True)

        weights = recon_audio.abs()
        if exists(prior):
            if prior.ndim == 2:
                prior = rearrange(prior, 'b n -> b n 1 1')
            elif prior.ndim == 3:
                prior = rearrange(prior, 'b n t -> b n 1 t')
            elif prior.ndim != 4:
                raise ValueError(f"Unsupported prior rank for mix closure: {prior.shape}")
            prior = prior.to(device=weights.device, dtype=weights.dtype).clamp_min(0.)
            weights = weights * prior

        closure_topk = self.exact_mix_closure_topk if topk is None else int(topk)
        if closure_topk > 0 and closure_topk < weights.shape[1]:
            topk_indices = weights.topk(k=closure_topk, dim=1).indices
            keep_mask = torch.zeros_like(weights)
            keep_mask.scatter_(1, topk_indices, 1.0)
            weights = weights * keep_mask

        weights_sum = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / recon_audio.shape[1])
        weights = torch.where(weights_sum > eps, weights / weights_sum.clamp_min(eps), uniform)

        corrected = recon_audio + weights * residual

        # Close the remaining numerical gap exactly by assigning it to the
        # dominant stem at each sample. The residual after weighted closure is
        # expected to be tiny; this step removes ISTFT roundoff from the sum.
        closure_residual = mix_audio - corrected.sum(dim=1)
        dominant_stem = corrected.abs().argmax(dim=1, keepdim=True)
        corrected = corrected.scatter_add(1, dominant_stem, closure_residual.unsqueeze(1))
        return corrected, weights, closure_residual

    def _apply_owned_calibrator(self, mix_audio, pred_audio, device, dual_components=None):
        eps = 1e-8
        stem_stft = self._audio_to_flattened_stft(pred_audio, device)
        mix_stft = self._audio_to_flattened_stft(mix_audio, device)

        stem_mag = stem_stft.abs()
        mix_mag = mix_stft.abs().clamp_min(eps)
        residual_stft = mix_stft - stem_stft.sum(dim=1, keepdim=True)
        residual_ratio = residual_stft.abs() / mix_mag

        owned_raw = stem_mag / stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
        peakiness = owned_raw / owned_raw.amax(dim=1, keepdim=True).clamp_min(eps)
        stem_to_mix = stem_mag / mix_mag

        agreement = torch.ones_like(owned_raw)
        if exists(dual_components):
            full_stft = dual_components.get('full_masked_stft', None)
            clean_stft = dual_components.get('clean_masked_stft', None)
            if exists(full_stft) and exists(clean_stft):
                full_mag = full_stft.abs()
                clean_mag = clean_stft.abs()
                agreement = 1. - (clean_mag - full_mag).abs() / full_mag.clamp_min(eps)
                agreement = agreement.clamp_(0., 1.)

        features = torch.stack(
            (
                owned_raw,
                peakiness,
                residual_ratio.expand_as(owned_raw),
                stem_to_mix,
                agreement,
            ),
            dim=2
        )
        batch_size, num_stems = features.shape[:2]
        features = rearrange(features, 'b n c f t -> (b n) c f t')

        fallback = stem_mag / stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
        base_owned = owned_raw * peakiness
        base_tau = self.owned_calibrator_threshold
        if base_tau > 0.:
            base_owned = torch.where(base_owned >= base_tau, base_owned, torch.zeros_like(base_owned))
        base_weights = self._normalize_owned_weights(
            base_owned.pow(self.owned_calibrator_gamma),
            fallback,
            eps=eps
        )

        logits = self.owned_calibrator(features)
        logits = rearrange(logits, '(b n) 1 f t -> b n f t', b=batch_size, n=num_stems)

        calibrator_scale = torch.tanh(self.owned_calibrator_scale).to(
            device=device, dtype=logits.dtype
        )
        # Keep the graph alive even if the screening branch produces NaNs.
        # Using a Python-side fallback here detaches the entire training path
        # once all separator weights are frozen.
        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        safe_scale = torch.nan_to_num(calibrator_scale, nan=0.0, posinf=0.0, neginf=0.0)

        delta = None
        gain = None
        if self.owned_calibrator_type in {'band_delta', 'temporal_delta'}:
            delta = torch.tanh(safe_logits) * safe_scale * self.owned_calibrator_delta_scale
            corrected_log_weights = torch.log(base_weights.clamp_min(eps)) + delta
            learned_weights = torch.softmax(corrected_log_weights, dim=1)
            weights = learned_weights / learned_weights.sum(dim=1, keepdim=True).clamp_min(eps)
            if self.training and self.owned_calibrator_train_fallback_mix > 0.0:
                mix_coeff = min(float(self.owned_calibrator_train_fallback_mix), 1.0)
                weights = (1.0 - mix_coeff) * weights + mix_coeff * base_weights
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            gain = torch.exp(torch.clamp(torch.tanh(safe_logits) * safe_scale, min=-6.0, max=6.0))
            owned = owned_raw * peakiness * gain

            tau = self.owned_calibrator_threshold
            if self.training and exists(self.owned_calibrator_train_threshold):
                tau = self.owned_calibrator_train_threshold
            if tau > 0.:
                if self.training:
                    soft_width = self.owned_calibrator_soft_threshold_width
                    soft_gate = torch.sigmoid((owned - tau) / soft_width)
                    owned = owned * soft_gate
                else:
                    owned = torch.where(owned >= tau, owned, torch.zeros_like(owned))

            owned_weights = owned.pow(self.owned_calibrator_gamma)
            if self.training and self.owned_calibrator_train_fallback_mix > 0.0:
                mixed_weights = owned_weights + self.owned_calibrator_train_fallback_mix * fallback
                weights = mixed_weights / mixed_weights.sum(dim=1, keepdim=True).clamp_min(eps)
            else:
                weights = self._normalize_owned_weights(
                    owned_weights,
                    fallback,
                    eps=eps
                )

        final_stft = stem_stft + (
            weights.to(stem_stft.dtype) * residual_stft
        ) * self.owned_calibrator_residual_scale
        refined_audio = self._flattened_stft_to_audio(final_stft, device, mix_audio.shape[-1])
        pre_closure_audio = refined_audio
        closure_weights = None
        closure_residual = None
        if self.use_exact_mix_closure:
            closure_prior = weights.mean(dim=(2, 3))
            refined_audio, closure_weights, closure_residual = self._apply_waveform_mix_closure(
                refined_audio,
                mix_audio,
                prior=closure_prior,
                topk=self.exact_mix_closure_topk,
            )

        aux = {
            'calibrated_recon': refined_audio,
            'calibrated_masked_stft': final_stft,
            'owned_calibrator_pre_closure_recon': pre_closure_audio,
            'owned_calibrator_weights': weights,
            'owned_calibrator_base_weights': base_weights,
            'owned_calibrator_gain': gain,
            'owned_calibrator_delta': delta,
            'owned_calibrator_closure_weights': closure_weights,
            'owned_calibrator_closure_residual': closure_residual,
        }
        return refined_audio, aux

    def forward(
            self,
            raw_audio,
            target=None,
            active_stem_ids=None,
            aux_target=None,
            return_loss_breakdown=False,
            return_inference_dual_components=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        mix_audio = raw_audio

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2),\
            ('stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2).'
             ' also need to be False if mono (channel dimension of 1)')

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(
                raw_audio,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=True
            )
        except:
            stft_repr = torch.stft(
                raw_audio.cpu() if x_is_mps else raw_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            time_screening_selector = self.time_screening_selectors[i]
            if self.use_torch_checkpoint and not isinstance(time_screening_selector, ZeroModule):
                x = x + checkpoint(time_screening_selector, x, use_reentrant=False)
            else:
                x = x + time_screening_selector(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        if len(self.delta_trunk_layers) > 0:
            delta_input = x
            delta_x = x

            for transformer_block in self.delta_trunk_layers:

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    delta_x, ft_ps = pack([delta_x], 'b * d')
                    if self.use_torch_checkpoint:
                        delta_x = checkpoint(linear_transformer, delta_x, use_reentrant=False)
                    else:
                        delta_x = linear_transformer(delta_x)
                    delta_x, = unpack(delta_x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                delta_x = rearrange(delta_x, 'b t f d -> b f t d')
                delta_x, ps = pack([delta_x], '* t d')

                if self.use_torch_checkpoint:
                    delta_x = checkpoint(time_transformer, delta_x, use_reentrant=False)
                else:
                    delta_x = time_transformer(delta_x)

                delta_x, = unpack(delta_x, ps, '* t d')
                delta_x = rearrange(delta_x, 'b f t d -> b t f d')
                delta_x, ps = pack([delta_x], '* f d')

                if self.use_torch_checkpoint:
                    delta_x = checkpoint(freq_transformer, delta_x, use_reentrant=False)
                else:
                    delta_x = freq_transformer(delta_x)

                delta_x, = unpack(delta_x, ps, '* f d')

            delta_scale = self.delta_trunk_scale.to(device=device, dtype=delta_x.dtype)
            x = delta_input + (delta_x - delta_input) * delta_scale

        if not isinstance(self.final_time_screening_selector, ZeroModule):
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = x + checkpoint(self.final_time_screening_selector, x, use_reentrant=False)
            else:
                x = x + self.final_time_screening_selector(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')

        x_pre_final_screen = x
        x = self.final_norm(x)

        def decode_with_heads(
                x_repr,
                heads,
                residual_routing_source=None,
                correction_heads=None,
                clean_heads=None,
                gate_heads=None,
                clean_band_selectors=None,
                refiner_heads=None,
                selected_stem_ids=None,
                return_pre_correction_recon=False,
                return_components=False,
        ):
            num_selected_stems = len(heads)

            if self.use_torch_checkpoint:
                mask = torch.stack([checkpoint(fn, x_repr, use_reentrant=False) for fn in heads], dim=1)
            else:
                mask = torch.stack([fn(x_repr) for fn in heads], dim=1)

            full_mask = mask
            clean_mask = None
            gate = None
            has_dual_heads = False
            output_mask = full_mask

            if (
                clean_heads is not None and
                gate_heads is not None and
                self.use_dual_output_heads and
                len(clean_heads) == num_selected_stems and
                len(gate_heads) == num_selected_stems and
                exists(self.dual_gate_bias) and
                exists(selected_stem_ids)
            ):
                clean_x_reprs = None
                if (
                    clean_band_selectors is not None and
                    self.use_clean_band_screening and
                    len(clean_band_selectors) == num_selected_stems
                ):
                    clean_x_reprs = []
                    clean_x_repr_packed, clean_x_repr_ps = pack([x_repr], '* f d')
                    for band_selector in clean_band_selectors:
                        if self.use_torch_checkpoint:
                            screened_clean_x = clean_x_repr_packed + checkpoint(
                                band_selector,
                                clean_x_repr_packed,
                                use_reentrant=False
                            )
                        else:
                            screened_clean_x = clean_x_repr_packed + band_selector(clean_x_repr_packed)
                        screened_clean_x, = unpack(screened_clean_x, clean_x_repr_ps, '* f d')
                        clean_x_reprs.append(screened_clean_x)

                if self.use_torch_checkpoint:
                    clean_mask = torch.stack(
                        [
                            checkpoint(
                                fn,
                                clean_x_reprs[idx] if clean_x_reprs is not None else x_repr,
                                use_reentrant=False
                            )
                            for idx, fn in enumerate(clean_heads)
                        ],
                        dim=1
                    )
                    gate_logits = torch.stack(
                        [checkpoint(fn, x_repr, use_reentrant=False) for fn in gate_heads],
                        dim=1
                    )
                else:
                    clean_mask = torch.stack(
                        [
                            fn(clean_x_reprs[idx] if clean_x_reprs is not None else x_repr)
                            for idx, fn in enumerate(clean_heads)
                        ],
                        dim=1
                    )
                    gate_logits = torch.stack([fn(x_repr) for fn in gate_heads], dim=1)

                gate_bias = self.dual_gate_bias[selected_stem_ids].to(
                    device=device, dtype=gate_logits.dtype
                ).view(1, num_selected_stems, 1, 1)
                gate = torch.sigmoid(gate_logits + gate_bias)
                output_mask = gate * clean_mask + (1. - gate) * full_mask
                has_dual_heads = True

            if (
                correction_heads is not None and
                self.use_post_mask_mlp_correction and
                len(correction_heads) == num_selected_stems and
                exists(self.post_mask_correction_scale) and
                exists(selected_stem_ids)
            ):
                if self.use_torch_checkpoint:
                    correction = torch.stack(
                        [checkpoint(fn, x_repr, use_reentrant=False) for fn in correction_heads],
                        dim=1
                    )
                else:
                    correction = torch.stack([fn(x_repr) for fn in correction_heads], dim=1)

                correction_scale = self.post_mask_correction_scale[selected_stem_ids].to(
                    device=device, dtype=mask.dtype
                ).view(1, num_selected_stems, 1, 1)
                corrected_mask = full_mask + correction * correction_scale
                has_correction = True
            else:
                corrected_mask = full_mask
                has_correction = False

            if not has_dual_heads:
                output_mask = corrected_mask

            def compute_router_weights():
                if (
                    residual_routing_source is None or
                    not self.use_residual_add_back_router or
                    len(self.residual_router_estimators) != num_selected_stems
                ):
                    return None

                if self.use_torch_checkpoint:
                    router_raw = torch.stack(
                        [checkpoint(fn, residual_routing_source, use_reentrant=False)
                         for fn in self.residual_router_estimators],
                        dim=1
                    )
                else:
                    router_raw = torch.stack(
                        [fn(residual_routing_source) for fn in self.residual_router_estimators],
                        dim=1
                    )
                router_raw = rearrange(router_raw, 'b n t (f c) -> b n f t c', c=2)
                router_scores = router_raw.pow(2).sum(dim=-1)
                router_weights = torch.softmax(router_scores, dim=1)
                router_priors = self.residual_router_stem_priors.to(
                    device=device, dtype=router_weights.dtype
                ).view(1, -1, 1, 1)
                router_weights = router_weights * router_priors
                router_weights = router_weights / router_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                return router_weights

            def compute_residual_refiner_weights(residual_stft):
                if (
                    not self.use_residual_only_refiner or
                    refiner_heads is None or
                    len(refiner_heads) != num_selected_stems
                ):
                    return None

                residual_real = torch.view_as_real(rearrange(residual_stft, 'b 1 f t -> b f t'))
                residual_repr = rearrange(residual_real, 'b f t c -> b t (f c)')
                if self.use_torch_checkpoint:
                    residual_repr = checkpoint(self.band_split, residual_repr, use_reentrant=False)
                else:
                    residual_repr = self.band_split(residual_repr)

                if self.use_torch_checkpoint:
                    refiner_raw = torch.stack(
                        [checkpoint(fn, residual_repr, use_reentrant=False)
                         for fn in refiner_heads],
                        dim=1
                    )
                else:
                    refiner_raw = torch.stack(
                        [fn(residual_repr) for fn in refiner_heads],
                        dim=1
                    )

                refiner_raw = rearrange(refiner_raw, 'b n t (f c) -> b n f t c', c=2)
                refiner_scores = refiner_raw.pow(2).sum(dim=-1)
                refiner_weights = torch.softmax(refiner_scores, dim=1)
                refiner_priors = self.residual_refiner_stem_priors.to(
                    device=device, dtype=refiner_weights.dtype
                ).view(1, -1, 1, 1)
                refiner_weights = refiner_weights * refiner_priors
                refiner_weights = refiner_weights / refiner_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                return refiner_weights

            def apply_masks(mask_tensor, use_router=True):
                mask_tensor = rearrange(mask_tensor, 'b n t (f c) -> b n f t c', c=2)
                masked_stft_local = base_stft_repr * torch.view_as_complex(mask_tensor)

                if use_router:
                    residual_stft = base_stft_repr - masked_stft_local.sum(dim=1, keepdim=True)
                    refiner_weights = compute_residual_refiner_weights(residual_stft)
                    if refiner_weights is not None:
                        masked_stft_local = masked_stft_local + (
                            refiner_weights.to(masked_stft_local.dtype) * residual_stft
                        ) * self.residual_refiner_scale
                    else:
                        router_weights = compute_router_weights()
                        if router_weights is not None:
                            masked_stft_local = masked_stft_local + (
                                router_weights.to(masked_stft_local.dtype) * residual_stft
                            ) * self.residual_router_scale
                return masked_stft_local

            def stft_to_audio(masked_stft_local):
                masked_stft_local = rearrange(masked_stft_local, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
                if self.zero_dc:
                    masked_stft_local = masked_stft_local.index_fill(1, tensor(0, device=device), 0.)

                try:
                    recon_local = torch.istft(
                        masked_stft_local,
                        **self.stft_kwargs,
                        window=stft_window,
                        return_complex=False,
                        length=raw_audio.shape[-1]
                    )
                except:
                    recon_local = torch.istft(
                        masked_stft_local.cpu() if x_is_mps else masked_stft_local,
                        **self.stft_kwargs,
                        window=stft_window.cpu() if x_is_mps else stft_window,
                        return_complex=False,
                        length=raw_audio.shape[-1]
                    ).to(device)

                recon_local = rearrange(
                    recon_local,
                    '(b n s) t -> b n s t',
                    s=self.audio_channels,
                    n=num_selected_stems
                )
                return recon_local

            def compute_partition_statistics(proposal_stft):
                eps = 1e-8
                proposal_mag = proposal_stft.abs()
                mix_mag = base_stft_repr.abs().clamp_min(eps)
                owned_raw = proposal_mag / proposal_mag.sum(dim=1, keepdim=True).clamp_min(eps)
                peakiness = owned_raw / owned_raw.amax(dim=1, keepdim=True).clamp_min(eps)
                base_owned = owned_raw * peakiness
                fallback = proposal_mag + 1.0
                base_weights = self._normalize_owned_weights(base_owned.pow(2.0), fallback, eps=eps)
                residual_stft = base_stft_repr - proposal_stft.sum(dim=1, keepdim=True)
                residual_ratio = residual_stft.abs() / mix_mag
                stem_to_mix = proposal_mag / mix_mag
                return {
                    'base_weights': base_weights,
                    'owned_raw': owned_raw,
                    'peakiness': peakiness,
                    'residual_ratio': residual_ratio,
                    'stem_to_mix': stem_to_mix,
                }

            def decode_with_partition():
                assert exists(self.partition_bucket_bias)
                assert num_selected_stems == self.num_stems and (
                    selected_stem_ids == list(range(self.num_stems))
                ), "partition output head currently requires all stems to be decoded together"

                proposal_stft = apply_masks(full_mask, use_router=False)
                partition_stats = compute_partition_statistics(proposal_stft)
                base_weights = partition_stats['base_weights']

                if self.output_head_type == 'partition':
                    assert exists(self.partition_source_delta_head)
                    assert exists(self.partition_bucket_head)
                    assert exists(self.partition_refine_delta_head)
                    source_delta = self.partition_source_delta_head(x_repr)
                    source_delta = rearrange(source_delta, 'b t f n -> b n f t')
                    bucket_logits = self.partition_bucket_head(x_repr)
                    bucket_logits = rearrange(bucket_logits, 'b t f 1 -> b 1 f t')
                    refine_delta = self.partition_refine_delta_head(x_repr)
                    refine_delta = rearrange(refine_delta, 'b t f n -> b n f t')
                else:
                    assert self.output_head_type == 'partition_screening'
                    assert exists(self.partition_screening_head)

                    residual_ratio = partition_stats['residual_ratio']
                    owned_raw = partition_stats['owned_raw']
                    peakiness = partition_stats['peakiness']
                    stem_to_mix = partition_stats['stem_to_mix']

                    source_features = torch.stack(
                        (
                            base_weights,
                            owned_raw,
                            peakiness,
                            stem_to_mix,
                            residual_ratio.expand_as(base_weights),
                        ),
                        dim=2,
                    )

                    bucket_features = torch.stack(
                        (
                            residual_ratio.squeeze(1),
                            1.0 - base_weights.amax(dim=1),
                            1.0 - owned_raw.amax(dim=1),
                            1.0 - stem_to_mix.amax(dim=1),
                            torch.ones_like(residual_ratio.squeeze(1)),
                        ),
                        dim=1,
                    )

                    source_delta, bucket_logits, refine_delta = self.partition_screening_head(
                        source_features,
                        bucket_features,
                    )

                eps = 1e-8
                source_logits = torch.log(base_weights.clamp_min(eps)) + source_delta
                bucket_logits = bucket_logits + self.partition_bucket_bias.to(
                    device=device, dtype=bucket_logits.dtype
                ).view(1, 1, 1, 1)
                ownership_logits = torch.cat((source_logits, bucket_logits), dim=1)
                ownership = torch.softmax(ownership_logits, dim=1)
                source_partition = ownership[:, :num_selected_stems]
                residual_bucket = ownership[:, num_selected_stems:num_selected_stems + 1]

                refine_logits = torch.log(base_weights.clamp_min(eps)) + refine_delta
                refine_weights = torch.softmax(refine_logits, dim=1)

                final_masked_stft = (
                    source_partition.to(base_stft_repr.dtype) * base_stft_repr
                    + refine_weights.to(base_stft_repr.dtype) * residual_bucket.to(base_stft_repr.dtype) * base_stft_repr
                )

                final_recon = stft_to_audio(final_masked_stft)
                closed_recon, closure_weights, closure_residual = self._apply_waveform_mix_closure(
                    final_recon,
                    mix_audio,
                )

                if return_components:
                    components = {
                        'proposal_recon': stft_to_audio(proposal_stft),
                        'final_recon': closed_recon,
                        'proposal_masked_stft': proposal_stft,
                        'final_masked_stft': final_masked_stft,
                        'mix_stft': base_stft_repr,
                        'partition_base_weights': base_weights,
                        'partition_source_weights': source_partition,
                        'partition_residual_bucket': residual_bucket,
                        'partition_refine_weights': refine_weights,
                        'partition_waveform_closure_weights': closure_weights,
                        'partition_waveform_closure_residual': closure_residual,
                    }
                    if self.output_head_type == 'partition_screening':
                        components.update(
                            {
                                'partition_screening_source_delta': source_delta,
                                'partition_screening_bucket_logits': bucket_logits,
                                'partition_screening_refine_delta': refine_delta,
                            }
                        )
                    return components['final_recon'], components

                return closed_recon

            if self.output_head_type in {'partition', 'partition_screening'}:
                return decode_with_partition()

            if return_components and has_dual_heads:
                full_masked_stft = apply_masks(full_mask, use_router=False)
                clean_masked_stft = apply_masks(clean_mask, use_router=False)
                base_masked_stft = apply_masks(output_mask, use_router=False)
                final_masked_stft = apply_masks(output_mask, use_router=True)

                components = {
                    'full_recon': stft_to_audio(full_masked_stft),
                    'clean_recon': stft_to_audio(clean_masked_stft),
                    'base_recon': stft_to_audio(base_masked_stft),
                    'final_recon': stft_to_audio(final_masked_stft),
                    'full_masked_stft': full_masked_stft,
                    'clean_masked_stft': clean_masked_stft,
                    'base_masked_stft': base_masked_stft,
                    'final_masked_stft': final_masked_stft,
                    'mix_stft': base_stft_repr,
                    'gate': gate,
                }
                return components['final_recon'], components

            if return_pre_correction_recon and has_correction:
                base_recon = stft_to_audio(apply_masks(full_mask))
                recon = stft_to_audio(apply_masks(corrected_mask))
                return recon, base_recon

            recon = stft_to_audio(apply_masks(output_mask))
            return recon

        if active_stem_ids is None:
            heads = self.mask_estimators
            correction_heads = (
                self.post_mask_correction_estimators
                if self.use_post_mask_mlp_correction else None
            )
            clean_heads = (
                self.clean_mask_estimators
                if self.use_dual_output_heads else None
            )
            gate_heads = (
                self.dual_gate_estimators
                if self.use_dual_output_heads else None
            )
            clean_band_selectors = (
                self.clean_band_screening_selectors
                if self.use_clean_band_screening else None
            )
            refiner_heads = (
                self.residual_refiner_estimators
                if self.use_residual_only_refiner else None
            )
            stem_ids = list(range(len(self.mask_estimators)))
        else:
            heads = [self.mask_estimators[i] for i in active_stem_ids]
            correction_heads = (
                [self.post_mask_correction_estimators[i] for i in active_stem_ids]
                if self.use_post_mask_mlp_correction else None
            )
            clean_heads = (
                [self.clean_mask_estimators[i] for i in active_stem_ids]
                if self.use_dual_output_heads else None
            )
            gate_heads = (
                [self.dual_gate_estimators[i] for i in active_stem_ids]
                if self.use_dual_output_heads else None
            )
            clean_band_selectors = (
                [self.clean_band_screening_selectors[i] for i in active_stem_ids]
                if self.use_clean_band_screening else None
            )
            refiner_heads = (
                [self.residual_refiner_estimators[i] for i in active_stem_ids]
                if self.use_residual_only_refiner else None
            )
            stem_ids = active_stem_ids

        num_stems = len(heads)
        base_stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        base_stft_repr = torch.view_as_complex(base_stft_repr)

        residual_routing_source = x_pre_final_screen if (
            active_stem_ids is None and self.use_residual_add_back_router
        ) else None
        use_post_mask_delta_supervision = (
            self.use_post_mask_mlp_correction and
            (
                self.post_mask_delta_loss_weight > 0. or
                self.post_mask_delta_consistency_loss_weight > 0.
            )
        )
        use_dual_head_supervision = self.use_dual_output_heads and exists(target)
        use_inference_dual_components = (
            self.use_dual_output_heads and
            return_inference_dual_components and
            not exists(target)
        )
        dual_components = None
        partition_components = None
        calibrator_aux = None
        if self.output_head_type in {'partition', 'partition_screening'} and exists(target):
            recon_audio, partition_components = decode_with_heads(
                x,
                heads,
                residual_routing_source=residual_routing_source,
                correction_heads=correction_heads,
                clean_heads=clean_heads,
                gate_heads=gate_heads,
                clean_band_selectors=clean_band_selectors,
                refiner_heads=refiner_heads,
                selected_stem_ids=stem_ids,
                return_components=True,
            )
            pre_correction_recon_audio = None
        elif use_dual_head_supervision or use_inference_dual_components:
            recon_audio, dual_components = decode_with_heads(
                x,
                heads,
                residual_routing_source=residual_routing_source,
                correction_heads=correction_heads,
                clean_heads=clean_heads,
                gate_heads=gate_heads,
                clean_band_selectors=clean_band_selectors,
                refiner_heads=refiner_heads,
                selected_stem_ids=stem_ids,
                return_components=True,
            )
            pre_correction_recon_audio = None
        elif use_post_mask_delta_supervision and exists(target):
            recon_audio, pre_correction_recon_audio = decode_with_heads(
                x,
                heads,
                residual_routing_source=residual_routing_source,
                correction_heads=correction_heads,
                clean_heads=clean_heads,
                gate_heads=gate_heads,
                clean_band_selectors=clean_band_selectors,
                refiner_heads=refiner_heads,
                selected_stem_ids=stem_ids,
                return_pre_correction_recon=True,
            )
        else:
            recon_audio = decode_with_heads(
                x,
                heads,
                residual_routing_source=residual_routing_source,
                correction_heads=correction_heads,
                clean_heads=clean_heads,
                gate_heads=gate_heads,
                clean_band_selectors=clean_band_selectors,
                refiner_heads=refiner_heads,
                selected_stem_ids=stem_ids,
            )
            pre_correction_recon_audio = None

        if self.use_owned_calibrator:
            recon_audio, calibrator_aux = self._apply_owned_calibrator(
                mix_audio=mix_audio,
                pred_audio=recon_audio,
                device=device,
                dual_components=dual_components,
            )
            if exists(dual_components):
                dual_components.update(calibrator_aux)

        if not exists(target):
            if return_inference_dual_components and exists(dual_components):
                return recon_audio, dual_components
            return recon_audio

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        target_sel = target[:, stem_ids]
        stem_loss_weights = self.stem_loss_weights[stem_ids].to(device=device, dtype=recon_audio.dtype)
        stem_weight_sum = stem_loss_weights.sum().clamp_min(1e-8)

        target_audio_packed = rearrange(target_sel, 'b n s t -> (b n s) t')
        try:
            target_stft = torch.stft(
                target_audio_packed,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=True
            )
        except:
            target_stft = torch.stft(
                target_audio_packed.cpu() if x_is_mps else target_audio_packed,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)

        target_stft = rearrange(
            target_stft,
            '(b n s) f t -> b n (f s) t',
            b=target_sel.shape[0],
            n=num_stems,
            s=self.audio_channels
        )

        stem_l1_loss = torch.abs(recon_audio - target_sel).mean(dim=(0, 2, 3))
        loss = (stem_l1_loss * stem_loss_weights).sum() / stem_weight_sum

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, 'b n s t -> (b n s) t'),**res_stft_kwargs)
            target_Y = torch.stft(rearrange(target_sel, 'b n s t -> (b n s) t'),**res_stft_kwargs)

            stem_stft_loss = torch.abs(recon_Y - target_Y)
            stem_stft_loss = rearrange(
                stem_stft_loss,
                '(b n s) f t -> b n s f t',
                b=recon_audio.shape[0],
                n=num_stems,
                s=self.audio_channels
            ).mean(dim=(0, 2, 3, 4))
            multi_stft_resolution_loss = multi_stft_resolution_loss + (
                (stem_stft_loss * stem_loss_weights).sum() / stem_weight_sum
            )

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        mix_consistency_loss = 0.
        complement_loss = 0.
        if self.mix_consistency_loss_weight > 0. and num_stems == self.num_stems:
            mix_consistency_recon_audio = recon_audio
            if exists(calibrator_aux):
                pre_closure_recon = calibrator_aux.get('owned_calibrator_pre_closure_recon', None)
                if exists(pre_closure_recon):
                    mix_consistency_recon_audio = pre_closure_recon
            pred_mix = mix_consistency_recon_audio.sum(dim=1)
            target_mix = target_sel.sum(dim=1)
            mix_consistency_loss = F.l1_loss(pred_mix, target_mix)
            complement_weights = self.stem_complement_loss_weights.to(
                device=device, dtype=recon_audio.dtype
            )
            if torch.any(complement_weights > 0):
                target_mix_expanded = target_mix.unsqueeze(1)
                pred_non_target = recon_audio.sum(dim=1, keepdim=True) - recon_audio
                pred_from_complement = target_mix_expanded - pred_non_target
                stem_complement_l1 = torch.abs(pred_from_complement - target_sel).mean(dim=(0, 2, 3))
                complement_loss = (stem_complement_l1 * complement_weights).sum()

        weighted_mix_consistency_loss = mix_consistency_loss * self.mix_consistency_loss_weight
        weighted_complement_loss = complement_loss

        partition_refine_supervision_loss = 0.
        if (
            exists(partition_components)
            and self.partition_refine_supervision_loss_weight > 0.
        ):
            eps = 1e-8
            mix_stft = partition_components['mix_stft']
            source_partition = partition_components['partition_source_weights']
            residual_bucket = partition_components['partition_residual_bucket']
            refine_weights = partition_components['partition_refine_weights']
            fallback_weights = partition_components['partition_base_weights']

            pre_partition_stft = source_partition.to(mix_stft.dtype) * mix_stft
            bucket_stft = residual_bucket.to(mix_stft.dtype) * mix_stft

            target_mag = target_stft.abs()
            pre_partition_mag = pre_partition_stft.abs()
            bucket_mag = bucket_stft.abs().squeeze(1)

            deficit = F.relu(target_mag - pre_partition_mag)
            deficit_sum = deficit.sum(dim=1, keepdim=True)
            routing_target = torch.where(
                deficit_sum > eps,
                deficit / deficit_sum.clamp_min(eps),
                fallback_weights
            )

            refine_log_probs = torch.log(refine_weights.clamp_min(eps))
            routing_ce = -(routing_target * refine_log_probs).sum(dim=1)
            partition_refine_supervision_loss = (
                routing_ce * bucket_mag
            ).sum() / bucket_mag.sum().clamp_min(eps)

        weighted_partition_refine_supervision_loss = (
            partition_refine_supervision_loss * self.partition_refine_supervision_loss_weight
        )

        dual_full_head_loss = 0.
        dual_clean_owned_loss = 0.
        dual_clean_forbidden_loss = 0.
        if exists(dual_components):
            full_recon_audio = dual_components['full_recon']
            clean_masked_stft = dual_components['clean_masked_stft']

            full_head_l1 = torch.abs(full_recon_audio - target_sel).mean(dim=(0, 2, 3))
            dual_full_head_loss = (full_head_l1 * stem_loss_weights).sum() / stem_weight_sum

            target_mag = target_stft.abs()
            clean_mag = clean_masked_stft.abs()
            total_mag = target_mag.sum(dim=1, keepdim=True).clamp_min(1e-8)
            ownership_ratio = target_mag / total_mag

            stem_weight_view = stem_loss_weights.view(1, num_stems, 1, 1)

            owned_mask = ownership_ratio >= self.dual_clean_owned_threshold
            if owned_mask.any():
                owned_error = torch.abs(clean_mag - target_mag) * owned_mask.to(clean_mag.dtype)
                dual_clean_owned_loss = (
                    owned_error * stem_weight_view
                ).sum() / (
                    owned_mask.to(clean_mag.dtype) * stem_weight_view
                ).sum().clamp_min(1e-8)

            forbidden_mask = ownership_ratio <= self.dual_clean_forbidden_threshold
            if forbidden_mask.any():
                forbidden_error = clean_mag * forbidden_mask.to(clean_mag.dtype)
                dual_clean_forbidden_loss = (
                    forbidden_error * stem_weight_view
                ).sum() / (
                    forbidden_mask.to(clean_mag.dtype) * stem_weight_view
                ).sum().clamp_min(1e-8)

        weighted_dual_full_head_loss = dual_full_head_loss * self.dual_full_loss_weight
        weighted_dual_clean_owned_loss = dual_clean_owned_loss * self.dual_clean_owned_loss_weight
        weighted_dual_clean_forbidden_loss = (
            dual_clean_forbidden_loss * self.dual_clean_forbidden_loss_weight
        )

        post_mask_delta_loss = 0.
        post_mask_delta_consistency_loss = 0.
        if exists(pre_correction_recon_audio):
            detached_pre_correction_recon = pre_correction_recon_audio.detach()
            pred_delta = recon_audio - detached_pre_correction_recon
            target_delta = target_sel - detached_pre_correction_recon
            stem_delta_l1 = torch.abs(pred_delta - target_delta).mean(dim=(0, 2, 3))
            post_mask_delta_loss = (stem_delta_l1 * stem_loss_weights).sum() / stem_weight_sum

            target_mix = target_sel.sum(dim=1)
            pred_mix_delta = pred_delta.sum(dim=1)
            target_mix_delta = target_mix - detached_pre_correction_recon.sum(dim=1)
            post_mask_delta_consistency_loss = F.l1_loss(pred_mix_delta, target_mix_delta)

        weighted_post_mask_delta_loss = post_mask_delta_loss * self.post_mask_delta_loss_weight
        weighted_post_mask_delta_consistency_loss = (
            post_mask_delta_consistency_loss * self.post_mask_delta_consistency_loss_weight
        )

        aux_loss = 0.
        if (
            self.pre_final_aux_mask_estimator is not None and
            exists(aux_target) and
            active_stem_ids is None
        ):
            if aux_target.ndim == 2:
                aux_target = rearrange(aux_target, '... t -> ... 1 t')
            if aux_target.ndim == 3:
                aux_target = rearrange(aux_target, 'b s t -> b 1 s t')

            aux_target = aux_target[..., :recon_audio.shape[-1]]
            aux_recon_audio = decode_with_heads(x_pre_final_screen, [self.pre_final_aux_mask_estimator])
            aux_l1_loss = F.l1_loss(aux_recon_audio, aux_target)

            aux_multi_stft_loss = 0.
            for window_size in self.multi_stft_resolutions_window_sizes:
                res_stft_kwargs = dict(
                    n_fft=max(window_size, self.multi_stft_n_fft),
                    win_length=window_size,
                    return_complex=True,
                    window=self.multi_stft_window_fn(window_size, device=device),
                    **self.multi_stft_kwargs,
                )
                aux_recon_Y = torch.stft(
                    rearrange(aux_recon_audio, 'b n s t -> (b n s) t'),
                    **res_stft_kwargs
                )
                aux_target_Y = torch.stft(
                    rearrange(aux_target, 'b n s t -> (b n s) t'),
                    **res_stft_kwargs
                )
                aux_multi_stft_loss = aux_multi_stft_loss + F.l1_loss(aux_recon_Y, aux_target_Y)

            aux_loss = aux_l1_loss + (aux_multi_stft_loss * self.multi_stft_resolution_loss_weight)

        weighted_aux_loss = aux_loss * self.pre_final_aux_head_loss_weight

        total_loss = (
            loss +
            weighted_multi_resolution_loss +
            weighted_mix_consistency_loss +
            weighted_complement_loss +
            weighted_partition_refine_supervision_loss +
            weighted_dual_full_head_loss +
            weighted_dual_clean_owned_loss +
            weighted_dual_clean_forbidden_loss +
            weighted_post_mask_delta_loss +
            weighted_post_mask_delta_consistency_loss +
            weighted_aux_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (
            loss,
            multi_stft_resolution_loss,
            mix_consistency_loss,
            complement_loss,
            partition_refine_supervision_loss,
            dual_full_head_loss,
            dual_clean_owned_loss,
            dual_clean_forbidden_loss,
            post_mask_delta_loss,
            post_mask_delta_consistency_loss,
            aux_loss,
        )
