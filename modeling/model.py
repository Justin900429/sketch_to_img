from typing import Any, Dict, List, Optional, Tuple, Union

import torch.fft as fft
from diffusers import UNet2DModel

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
import xformers.ops
from einops import rearrange
from torch.nn.functional import silu
from torchvision.models import ResNet34_Weights, resnet34
from xformers.ops import (
    MemoryEfficientAttentionFlashAttentionOp,
    memory_efficient_attention,
)

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        # w_pad, f_pad = int(w_pad), int(f_pad) # for flops calculation

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        print("loading flash attention fail.", str(e))

    return None


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
        eff_attn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (num_heads if num_heads is not None else out_channels // channels_per_head)
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.eff_attn = eff_attn

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
                .unbind(2)
            )
            if self.eff_attn:  # add mea for around 15% cost reduction
                q, k, v = map(
                    lambda t: rearrange(t, "(b h) d n -> b n h d", h=self.num_heads),
                    (q, k, v),
                )
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                out = memory_efficient_attention(q, k, v)
                a = rearrange(out, "b n h d -> (b h) d n", h=self.num_heads)
            else:
                w = AttentionOp.apply(q, k)
                a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion


class DhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        use_condition=False,
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        eff_attn=False,  # Add efficient attention
        use_free_U=False,  # Use FreeU
        b1=1.2,
        b2=1.4,
        s1=0.9,
        s2=0.2,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
            eff_attn=eff_attn,
        )
        self.use_condition = use_condition
        if use_condition:
            self.conditional_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.conditional_model.fc = Linear(
                in_features=self.conditional_model.fc.in_features,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
        # Mapping.
        self.label_dim = label_dim
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

        self.use_free_U = use_free_U
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(
        self,
        x,
        noise_labels,
        class_labels=None,
        conditional_imgs=None,
        augment_labels=None,
    ):
        # Mapping.
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        conditonal_feature = (
            None
            if not self.use_condition
            else self.conditional_model(
                torch.zeros_like(x) if conditional_imgs is None else conditional_imgs
            )
        )
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(
                    tmp.dtype
                )
            emb = emb + self.map_label(tmp)
        if self.use_condition:
            emb = emb + conditonal_feature
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for key, block in self.dec.items():
            half_len = x.shape[1] // 2
            if x.shape[1] != block.in_channels:
                pop_out = skips.pop()
                if self.use_free_U and "block0" in key:
                    x[:, :half_len] = x[:, :half_len] * self.b1
                    pop_out = Fourier_filter(pop_out, threshold=1, scale=self.s1)
                elif self.use_free_U and "block1" in key:
                    x[:, :half_len] = x[:, :half_len] * self.b2
                    pop_out = Fourier_filter(pop_out, threshold=1, scale=self.s2)
                x = torch.cat([x, pop_out], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


def build_diffuser_model(cfg):
    layer_scale = [cfg.MODEL.BASE_DIM * scale for scale in cfg.MODEL.LAYER_SCALE]
    return UNet2DModel(
        sample_size=cfg.TRAIN.IMAGE_SIZE,
        in_channels=cfg.MODEL.IN_CHANNELS,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        layers_per_block=cfg.MODEL.LAYERS_PER_BLOCK,
        block_out_channels=layer_scale,
        down_block_types=cfg.MODEL.DOWN_BLOCK_TYPE,
        up_block_types=cfg.MODEL.UP_BLOCK_TYPE,
    )


def build_model(cfg, use_free_U=False):
    model = DhariwalUNet(
        img_resolution=cfg.TRAIN.IMAGE_SIZE_WIDTH,
        in_channels=cfg.MODEL.IN_CHANNELS,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        label_dim=cfg.MODEL.LABEL_DIM,
        use_condition=cfg.MODEL.USE_CONDITION,
        augment_dim=0,
        model_channels=cfg.MODEL.BASE_DIM,
        channel_mult=cfg.MODEL.LAYER_SCALE,
        channel_mult_emb=4,
        num_blocks=cfg.MODEL.LAYERS_PER_BLOCK,
        dropout=0.10,
        label_dropout=0,
        eff_attn=True,
        use_free_U=use_free_U,
        b1=cfg.EVAL.FREE_U.b1,
        b2=cfg.EVAL.FREE_U.b2,
        s1=cfg.EVAL.FREE_U.s1,
        s2=cfg.EVAL.FREE_U.s2,
    )

    return model
