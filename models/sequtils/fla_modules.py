# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.
# https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/ops/triton/layernorm.py
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .fla_utils import input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    O,  # pointer to the gate
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * N
    Y += row * N
    O += row * N
    if HAS_RESIDUAL:
        RESIDUAL += row * N
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    # Swish output gate
    o = tl.load(O + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = y * o * tl.sigmoid(o)

    # Write output
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    o: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    if residual is not None:
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
    if bias is not None:
        assert bias.shape == (N,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((M,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((M,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    layer_norm_fwd_kernel[(M,)](
        x,
        o,
        y,
        weight,
        bias,
        residual,
        residual_out,
        mean,
        rstd,
        N,
        eps,
        is_rms_norm,
        BLOCK_N,
        residual is not None,
        residual_out is not None,
        weight is not None,
        bias is not None,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def layer_norm_bwd_kernel(
    X,  # pointer to the input
    O,  # pointer to the gate
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DO,  # pointer to the gate gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * N
    O += row_start * N
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * N
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * N
    DY += row_start * N
    DX += row_start * N
    DO += row_start * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        o = tl.load(O + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)

        y = xhat * w if HAS_WEIGHT else xhat
        if HAS_BIAS:
            y = y + b
        if RECOMPUTE_OUTPUT:
            tl.store(Y + cols, y, mask=mask)

        sigmoid_o = tl.sigmoid(o)
        do = dy * y * (sigmoid_o + o * sigmoid_o * (1 - sigmoid_o))
        dy = dy * o * sigmoid_o
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        tl.store(DO + cols, do, mask=mask)

        X += N
        O += N
        if HAS_DRESIDUAL:
            DRESIDUAL += N
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += N
        if RECOMPUTE_OUTPUT:
            Y += N
        DY += N
        DX += N
        DO += N
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    o: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    M, N = x.shape
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
    if bias is not None:
        assert bias.shape == (N,)
    # allocate output
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    do = (
        torch.empty_like(o)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = (
        torch.empty(M, N, dtype=dy.dtype, device=dy.device)
        if recompute_output
        else None
    )

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    dw = (
        torch.empty((sm_count, N), dtype=torch.float, device=weight.device)
        if weight is not None
        else None
    )
    db = (
        torch.empty((sm_count, N), dtype=torch.float, device=bias.device)
        if bias is not None
        else None
    )
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    layer_norm_bwd_kernel[grid](
        x,
        o,
        weight,
        bias,
        y,
        dy,
        dx,
        do,
        dw,
        db,
        dresidual,
        dresidual_in,
        mean,
        rstd,
        M,
        N,
        eps,
        rows_per_program,
        is_rms_norm,
        BLOCK_N,
        dresidual is not None,
        dresidual_in is not None,
        weight is not None,
        bias is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (
        (dx, do, dw, db, dresidual_in)
        if not recompute_output
        else (dx, do, dw, db, dresidual_in, y)
    )


class LayerNormSwishGateFn(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        o,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        o_shape_og = o.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x,
            o,
            weight,
            bias,
            eps,
            residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(residual_out, o, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x, o, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dw, db, dresidual_in = layer_norm_bwd(
            dy,
            x,
            o,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            do.reshape(ctx.o_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


class LayerNormSwishGateLinearFn(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        o_shape_og = o.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x,
            o,
            norm_weight,
            norm_bias,
            eps,
            residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = (
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        )
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(
            residual_out, o, norm_weight, norm_bias, linear_weight, mean, rstd
        )
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, o, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy,
            x,
            o,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            do.reshape(ctx.o_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_swish_gate_fn(
    x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6
):
    return LayerNormSwishGateFn.apply(
        x, o, weight, bias, residual, eps, prenorm, residual_in_fp32, False
    )


def rms_norm_swish_gate_fn(
    x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6
):
    return LayerNormSwishGateFn.apply(
        x, o, weight, bias, residual, eps, prenorm, residual_in_fp32, True
    )


def layer_norm_swish_gate_linear_fn(
    x,
    o,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6,
):
    return LayerNormSwishGateLinearFn.apply(
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        False,
    )


def rms_norm_swish_gate_linear_fn(
    x,
    o,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6,
):
    return LayerNormSwishGateLinearFn.apply(
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


class FusedLayerNormSwishGate(nn.Module):

    def __init__(
        self, hidden_size, elementwise_affine: bool = True, eps=1e-5
    ) -> FusedLayerNormSwishGate:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_swish_gate_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedRMSNormSwishGate(nn.Module):

    def __init__(
        self, hidden_size, elementwise_affine: bool = True, eps=1e-5
    ) -> FusedRMSNormSwishGate:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_swish_gate_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedLayerNormSwishGateLinear(nn.Module):

    def __init__(
        self, hidden_size, elementwise_affine: bool = True, eps=1e-5
    ) -> FusedLayerNormSwishGateLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self, x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False
    ):
        return layer_norm_swish_gate_linear_fn(
            x,
            o,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedRMSNormSwishGateLinear(nn.Module):

    def __init__(
        self, hidden_size, elementwise_affine: bool = True, eps=1e-5
    ) -> FusedRMSNormSwishGateLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self, x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False
    ):
        return rms_norm_swish_gate_linear_fn(
            x,
            o,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["N"],
)
@triton.jit
def l2norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    y = x * rstd
    # Write output
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["N"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
# @triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def l2norm_bwd_kernel(
    X,  # pointer to the input
    # Y, # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row

    # Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    # y = x * rstd
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    # dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
        M, N = x.shape
    assert x.stride(-1) == 1
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    l2norm_fwd_kernel[(M,)](
        x,
        y,
        x.stride(0),
        N,
        eps,
        # is_rms_norm,
        BLOCK_N,
        # residual is not None,
        # residual_out is not None,
        # bias is not None,
    )
    return y.reshape(x_shape_og)


def l2norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    # allocate output
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    l2norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        x.stride(0),
        N,
        eps,
        BLOCK_N,
    )
    return dx.reshape(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        # reshape input data into 2D tensor
        y = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy, *args):
        (x,) = ctx.saved_tensors
        dx = l2norm_bwd(x, dy, ctx.eps)
        return dx, None, None


def l2_norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Tri Dao, Yu Zhang, Songlin Yang.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .fla_utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

sigmoid_fwd_codestring = """
template <typename T> T sigmoid_fwd(T x) {
    return 1.0f / (1.0f + ::exp(-float(x)));
}
"""
sigmoid_bwd_codestring = """
template <typename T> T sigmoid_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - x_sigmoid);
}
"""

sigmoid_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_fwd_codestring)
sigmoid_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_bwd_codestring)


@torch.compiler.disable
def sigmoid_fwd(x):
    return sigmoid_fwd_jit_fn(x)


@torch.compiler.disable
def sigmoid_bwd(x, g):
    return sigmoid_bwd_jit_fn(x, g)


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def logsigmoid_fwd_kernel(
    x, y, temperature, T: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + tl.exp(-tl.abs(b_x))
    b_y = (b_m - tl.log(b_z)) / temperature
    tl.store(y + o_i, b_y.to(y.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def logsigmoid_bwd_kernel(
    x, dx, dy, temperature, T: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_dx = b_dy * (1.0 - tl.sigmoid(b_x)) / temperature
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)


def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(
        triton.cdiv(T, torch.cuda.get_device_properties(x.device).multi_processor_count)
    )
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[(triton.cdiv(T, B),)](
        x=x, y=y, temperature=temperature, T=T, D=D, B=B
    )
    return y


def logsigmoid_bwd(
    x: torch.Tensor, dy: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(
        triton.cdiv(T, torch.cuda.get_device_properties(x.device).multi_processor_count)
    )
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[(triton.cdiv(T, B),)](
        x=x, dx=dx, dy=dy, temperature=temperature, T=T, D=D, B=B
    )
    return dx


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, temperature):
        ctx.save_for_backward(
            x,
        )
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


def logsigmoid(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return LogSigmoidFunction.apply(x, temperature)


swish_fwd_codestring = """
template <typename T> T swish_fwd(T x) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(x) * x_sigmoid;
}
"""
swish_bwd_codestring = """
template <typename T> T swish_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - float(x) * x_sigmoid + float(x));
}
"""

swish_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_fwd_codestring)
swish_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_bwd_codestring)


@torch.compiler.disable
def swish_fwd(x):
    return swish_fwd_jit_fn(x)


@torch.compiler.disable
def swish_bwd(x, g):
    return swish_bwd_jit_fn(x, g)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply

# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.compile
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(
        dtype=y.dtype
    )


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.compile
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(
        dtype=x.dtype
    )


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.compile
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.compile
def sqrelu_fwd(x):
    r = F.relu(x.float())
    return (r * r).to(dtype=x.dtype)


@torch.compile
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x.float())).to(dtype=x.dtype)


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


sqrelu = SquaredReLUFunction.apply


swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""

swiglu_fwdbwd_codestring = """
template <typename T> T swiglu_fwdbwd(T x, T y, T g, T& dx, T& dy, T& z) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    float x_swish = float(x) * x_sigmoid;
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = x_swish * float(g);
    z = x_swish * float(y);
}
"""


swiglu_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(
    swiglu_bwd_codestring, num_outputs=2
)
swiglu_fwdbwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(
    swiglu_fwdbwd_codestring, num_outputs=3
)


@torch.compiler.disable
def swiglu_fwd(x, y):
    return swiglu_fwd_jit_fn(x, y)


@torch.compiler.disable
def swiglu_bwd(x, y, g):
    return swiglu_bwd_jit_fn(x, y, g)


@torch.compiler.disable
def swiglu_fwdbwd(x, y, g):
    return swiglu_fwdbwd_jit_fn(x, y, g)


@torch.compile
def swiglu_fwd_torch(x, y):
    return (F.silu(x.float()) * y).to(x.dtype)


@torch.compile
def swiglu_bwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    return dx.to(dtype), dy.to(dtype)


@torch.compile
def swiglu_fwdbwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    z = x_swish * y
    return dx.to(dtype), dy.to(dtype), z.to(dtype)


class SwiGLUFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \text{SwiGLU}(x, y) = swish(x) * y = \frac{x}{1 + \exp(-x)} * y
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        if torch.compiler.is_compiling() or isinstance(
            x, torch.distributed.tensor.DTensor
        ):
            return swiglu_fwd_torch(x, y)
        else:
            return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        if torch.compiler.is_compiling() or isinstance(
            x, torch.distributed.tensor.DTensor
        ):
            return swiglu_bwd_torch(x, y, dout)
        else:
            return swiglu_bwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(
                x, torch.distributed.tensor.DTensor
            ):
                z = swiglu_fwd_torch(x, y)
            else:
                z = swiglu_fwd(x, y)
        out = F.linear(z, weight, bias)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(
                x, torch.distributed.tensor.DTensor
            ):
                dx, dy, z = swiglu_fwdbwd_torch(x, y, dz)
            else:
                dx, dy, z = swiglu_fwdbwd(x, y, dz)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


swiglu = SwiGLUFunction.apply


swiglu_linear = SwiGLULinearFunction.apply


ACT2FN = {
    "relu": F.relu,
    "sigmoid": sigmoid,
    "logsigmoid": logsigmoid,
    "silu": swish,
    "swish": swish,
    "sqrelu": sqrelu,
    "gelu": fast_gelu_impl,
    "bias_gelu": bias_gelu_impl,
}


# -*- coding: utf-8 -*-

# from https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/convolution.py

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .fla_utils import checkpoint

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@checkpoint
def proj_then_conv1d(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor] = None,
    cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # We do matmul and transpose BLH -> HBL at the same time
    x = rearrange(
        proj_weight @ rearrange(x, "b t d -> d (b t)"),
        "d (b t) -> b d t",
        t=x.shape[-2],
    )

    if causal_conv1d_fn is None:
        raise ImportError(
            "`causal_conv1d_fn` is not available. Please install `causal-conv1d` first."
        )
    if cache is None:
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            activation="silu",
        ).transpose(1, 2)
    else:
        assert x.shape[-1] == 1, "Only support decoding with 1 token at a time for now"
        x = x.squeeze(-1)
        x = causal_conv1d_update(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            cache=cache,
            activation="silu",
        )
    return x


class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = "silu",
        use_fast_conv1d: Optional[bool] = True,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in [
                "silu",
                "swish",
            ], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if causal_conv1d_fn is None:
            if use_fast_conv1d:
                raise RuntimeError(
                    "Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel "
                    "or set `use_fast_conv1d` to False"
                )
            else:
                warnings.warn(
                    "The naive Pytorch verison is very slow in practice, "
                    "please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel",
                    category=ImportWarning,
                )
        self.use_fast_conv1d = use_fast_conv1d

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        if self.activation is not None:
            s += ", activation={activation}"
        if not self.use_fast_conv1d:
            s += ", use_fast_conv1d={use_fast_conv1d}"
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[batch_size, seq_len, hidden_size]`
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[batch_size, hidden_size, kernel_size]`.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[batch_size, hidden_size, kernel_size]`. Default: `False`.
            seq_idx (Optional[torch.Tensor]):
                Sequence index for each token. Used for varlen. Default: `None`.
                Shape: [batch_size, seq_len]
                Suppose a batch consists of two sequences with lengths 3 and 4, seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
        Returns:
            Tensor of shape `[batch_size, seq_len, hidden_size]`.
        """

        batch_size, _, hidden_size = x.shape
        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(batch_size, hidden_size, self.kernel_size[0])
        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)
        x = rearrange(x, "b t d -> b d t")
        # Update state (B D W)
        if cache is not None:
            cache.copy_(F.pad(x, (self.kernel_size[0] - x.shape[-1], 0)))
        if self.use_fast_conv1d:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., : x.shape[-1]]
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return rearrange(x, "b d t -> b t d"), cache

    def step(self, x: torch.Tensor, cache: torch.Tensor):
        assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = ACT2FN[self.activation](x).to(dtype=dtype)
        return x.unsqueeze(1), cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed
    filter of length max_len.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        hidden_size (int): The number of expected features in the input and output.
        max_len (int): The maximum sequence length.
    Returns:
        y: [batch_size, seq_len, hidden_size] tensor
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            hidden_size (int): The number of expected features in the input and output.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(
            torch.randn(self.hidden_size, max_len), requires_grad=True
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        hidden_size (int):
            The number of expected features in the input and output.
        max_len (int):
            The maximum sequence length.
        d_emb (Optional[int]):
            The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine).
            Defaults to 3.
        d_hidden (Optional[int]):
            The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (`PositionalEmbedding`): The positional embedding layer.
        mlp (`nn.Sequential`): The MLP that parameterizes the implicit filter.

    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        d_emb: int = 3,
        d_hidden: int = 16,
        **kwargs,
    ):
        """
        Long convolution with implicit filter parameterized by an MLP.


        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb

        assert (
            d_emb % 2 != 0 and d_emb >= 3
        ), "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, max_len)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, hidden_size),
        )

    def filter(self, seq_len: int, *args, **kwargs):
        k = self.mlp(self.pos_emb(seq_len))

        return k.transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)
