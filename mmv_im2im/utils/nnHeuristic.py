import numpy as np
import torch


def get_nnunet_plans(patch_size, spacing, modality="non-CT", min_size=8, vram_gb=None):
    """
    Full replication of nnU-Net heuristic for topology and normalization.

    Args:
        patch_size: List/tuple of spatial dimensions (e.g., [256, 256] or [128, 128, 128])
        spacing: Voxel spacing (e.g., [1.0, 1.0] or [1.0, 1.0, 5.0])
        modality: "CT" or "non-CT" (determines normalization)
        vram_gb: Available GPU memory in GB (default is 40 for A100-40GB).
    """
    dim = len(patch_size)
    cur_size = np.array(patch_size)
    cur_spacing = np.array(spacing)

    if vram_gb is None:
        if torch.cuda.is_available():
            # Get total memory of the current device in bytes, then convert to GB
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = total_memory_bytes / (1024**3)
            print(
                f"Detected GPU with {vram_gb:.1f} GB VRAM. Adapting network topology..."
            )
        else:
            print("CUDA not available. Defaulting to safe 12GB VRAM heuristic.")
            vram_gb = 12.0
    else:
        vram_gb = vram_gb

    strides = []
    kernels = []

    # TOPOLOGY HEURISTIC (Strides & Kernels)
    # The first layer always has stride 1
    strides.append([1] * dim)
    kernels.append([3] * dim)

    if vram_gb >= 38:  # Using 38 as a safe threshold for 40GB cards
        max_f = 512 if dim == 3 else 1024
    elif vram_gb >= 22:  # Safe threshold for 24GB cards
        max_f = 416 if dim == 3 else 768
    else:
        # Standard <16GB fallback
        max_f = 320 if dim == 3 else 512

    while True:
        # Determine which axes to downsample based on size and anisotropy
        # Rule: Only downsample an axis if it's > min_size AND (it's at the
        # highest resolution OR we have already balanced the resolution)
        target_spacing = np.min(cur_spacing)

        # Decide stride for each axis
        new_stride = []
        for i in range(dim):
            # Downsample if axis is large enough AND its spacing is 'close enough'
            # to the target high-res spacing (within a factor of 2)
            if cur_size[i] > min_size and cur_spacing[i] <= 2 * target_spacing:
                new_stride.append(2)
            else:
                new_stride.append(1)

        # Termination: if no axis can be downsampled anymore
        if all(s == 1 for s in new_stride):
            break

        strides.append(new_stride)
        kernels.append([3] * dim)

        # Update current state for next iteration
        cur_size = cur_size // np.array(new_stride)
        cur_spacing = cur_spacing * np.array(new_stride)

        # FILTER HEURISTIC
        num_layers = len(strides)

    filters = [min(32 * (2**i), max_f) for i in range(num_layers)]

    # 3. NORMALIZATION & INTENSITY HEURISTIC
    # nnU-Net uses different intensity strategies based on modality
    norm_plans = {}
    if modality.upper() == "CT":
        # CT uses Global Normalization (Clipped to percentiles)
        norm_plans = {
            "norm_name": "instance",
            "intensity_clipping": [0.5, 99.5],  # Percentiles
            "z_score_type": "global",
        }
    else:
        # MRI/Microscopy uses Per-Image Z-Score
        norm_plans = {
            "norm_name": "instance",
            "intensity_clipping": None,
            "z_score_type": "per_image",
        }

    return {
        "kernel_size": kernels,
        "strides": strides,
        "filters": filters,
        "upsample_kernel_size": strides[1:],  # MONAI DynUNet specific
        "norm_info": norm_plans,
    }
