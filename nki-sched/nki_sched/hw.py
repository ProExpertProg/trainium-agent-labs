"""Hardware description: tile caps, memory limits, and the ns-level instruction table.

Numbers marked [unverified] have not been confirmed against the installed NKI/hardware (PLAN §3.2)
and are overridable; tile caps come from nkibench / the nc_matmul documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    pmax: int = 128  # partition dim max (nl.tile_size.pmax)
    stationary_fmax: int = 128  # nl.tile_size.gemm_stationary_fmax
    moving_fmax: int = 512  # nl.tile_size.gemm_moving_fmax
    psum_bank_bytes: int = 2048  # per partition, one accumulation bank   [unverified]
    psum_banks: int = 8  # [unverified]
    sbuf_bytes_per_partition: int = 192 * 1024  # [unverified]


NC_DEFAULT = HardwareConfig()
# A tiny configuration so scalar-loop interpretation of whole schedules stays fast in tests.
NC_TINY = HardwareConfig(pmax=4, stationary_fmax=4, moving_fmax=8, psum_bank_bytes=8 * 4)

DTYPE_BYTES = {"f32": 4, "bf16": 2, "f16": 2}


class HardwareError(Exception):
    """a program (or one instruction) violates a hardware rule"""


def itemsize(dtype: str) -> int:
    """bytes per element; dtypes that follow an input ("like:x") are sized for f32, the worst case"""
    return DTYPE_BYTES.get(dtype, 4)
