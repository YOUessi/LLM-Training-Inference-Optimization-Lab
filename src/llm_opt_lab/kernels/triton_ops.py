"""CUDA-only Triton kernels: one program per row.

Import from GPU benchmarking code only. Importing this module requires torch,
triton, and a compatible backend; cpu_reference.py works without them.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, Y, stride_x0: tl.constexpr, stride_y0: tl.constexpr,
                    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x0 + cols, cols < N, other=0).to(tl.float32)
    squares = tl.where(cols < N, x * x, 0.)
    rstd = tl.rsqrt(tl.sum(squares, 0) / N + EPS)
    y = x * rstd
    tl.store(Y + row * stride_y0 + cols, y, cols < N)


@triton.jit
def _softmax_kernel(X, Y, stride_x0: tl.constexpr, stride_y0: tl.constexpr,
                    N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x0 + cols, cols < N, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, 0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, 0)
    tl.store(Y + row * stride_y0 + cols, numerator / denominator, cols < N)


def _check(x: torch.Tensor) -> None:
    if x.device.type != "cuda":
        raise ValueError("Triton experiments require CUDA tensors")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("expected a contiguous 2D row-major tensor")
    if not (1 <= x.shape[1] <= 16384):
        raise ValueError("column dimension must be 1..16384 for this starter implementation")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("supported dtypes: float16, float32")


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unweighted RMSNorm over last dimension, FP32 internal accumulation."""
    _check(x)
    if eps <= 0:
        raise ValueError("eps must be positive")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    _rmsnorm_kernel[(x.shape[0],)](
        x, out, x.stride(0), out.stride(0), x.shape[1], eps, block,
        num_warps=4 if block <= 4096 else 8,
    )
    return out


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Stable row-wise Softmax, FP32 internal accumulation."""
    _check(x)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    _softmax_kernel[(x.shape[0],)](
        x, out, x.stride(0), out.stride(0), x.shape[1], block,
        num_warps=4 if block <= 4096 else 8,
    )
    return out



@triton.jit
def _rmsnorm_grouped_kernel(
    X, Y,
    stride_x0: tl.constexpr, stride_y0: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr,
):
    """Independent rows in the same program; zero padded columns/rows are masked."""
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(
        X + rows[:, None] * stride_x0 + cols[None, :],
        mask=valid, other=0,
    ).to(tl.float32)
    denominator = tl.sum(tl.where(cols[None, :] < N, x * x, 0.0), axis=1)
    inv_rms = tl.rsqrt(denominator / N + EPS)
    out = x * inv_rms[:, None]
    tl.store(
        Y + rows[:, None] * stride_y0 + cols[None, :],
        out, mask=valid,
    )


def rmsnorm_grouped(
    x: torch.Tensor,
    eps: float = 1e-6,
    *,
    rows_per_program: int = 2,
    num_warps: int = 4,
) -> torch.Tensor:
    """Parameter-explicit grouped RMSNorm for experimentation, not dispatch policy."""
    _check(x)
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be positive and finite")
    if rows_per_program not in (1, 2, 4, 8, 16):
        raise ValueError("rows_per_program must be one of 1,2,4,8,16")
    if num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be one of 1,2,4,8,16")
    block = triton.next_power_of_2(x.shape[1])
    if block * rows_per_program > 32768:
        raise ValueError("too many elements in one program")
    out = torch.empty_like(x)
    _rmsnorm_grouped_kernel[
        (triton.cdiv(x.shape[0], rows_per_program),)
    ](
        x, out, x.stride(0), out.stride(0), x.shape[0], x.shape[1],
        eps, block, rows_per_program, num_warps=num_warps,
    )
    return out
