"""Device mesh utilities for single/multi-device setups."""

import jax
import numpy as np
from jax.sharding import Mesh


def create_mesh(
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
) -> Mesh:
    """Create a JAX device mesh.

    Args:
        tp: Tensor parallelism degree.
        dp: Data parallelism degree.
        ep: Expert parallelism degree (for MoE models).

    Returns:
        A JAX Mesh with named axes.

    Examples:
        Stage 1 (single device):  create_mesh(tp=1, dp=1)
        Stage 2 (TP on 4 TPUs):   create_mesh(tp=4, dp=1)
        Stage 2 (DP+TP on 8):     create_mesh(tp=4, dp=2)
        Stage 4 (EP+TP):          create_mesh(tp=2, dp=1, ep=4)
    """
    devices = jax.devices()
    total_needed = tp * dp * ep
    assert len(devices) >= total_needed, (
        f"Need {total_needed} devices (tp={tp}, dp={dp}, ep={ep}), "
        f"but only {len(devices)} available."
    )

    devices_to_use = devices[:total_needed]

    if ep > 1:
        # 3D mesh: (dp, ep, tp)
        device_array = np.array(devices_to_use).reshape(dp, ep, tp)
        return Mesh(device_array, axis_names=("data", "expert", "tensor"))
    else:
        # 2D mesh: (dp, tp)
        device_array = np.array(devices_to_use).reshape(dp, tp)
        return Mesh(device_array, axis_names=("data", "tensor"))
