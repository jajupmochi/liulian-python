"""
Data augmentation techniques for time series forecasting (PyTorch implementation).
Adapted from Time-Series-Library for pure torch.Tensor operations.
All functions expect input shape: (batch_size, seq_len, n_features)
"""

import torch
import numpy as np
from typing import Optional, Literal


def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """
    Add random Gaussian noise to time series.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        sigma: Standard deviation of Gaussian noise
        
    Returns:
        Augmented tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)  # batch_size=32, seq_len=96, features=7
        >>> x_aug = jitter(x, sigma=0.05)
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Scale time series by random factors per sample and feature.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        sigma: Standard deviation of scaling factor (mean=1.0)
        
    Returns:
        Scaled tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = scaling(x, sigma=0.2)
    """
    # Generate scaling factors: shape (batch_size, 1, n_features)
    scale_factors = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma + 1.0
    return x * scale_factors


def rotation(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly rotate (flip) features in time series.
    Useful for multivariate series where feature order doesn't matter.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        
    Returns:
        Rotated tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = rotation(x)
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    # For each sample, flip features or not
    flip_prob = 0.5
    flip_mask = torch.rand(batch_size, device=device) < flip_prob
    
    # Randomly permute features for samples that flip
    ret = x.clone()
    for i in range(batch_size):
        if flip_mask[i]:
            # Random permutation of feature dimension
            perm = torch.randperm(n_features, device=device)
            ret[i] = ret[i, :, perm]
    
    return ret


def permutation(
    x: torch.Tensor, 
    max_segments: int = 5, 
    seg_mode: Literal["equal", "random"] = "equal"
) -> torch.Tensor:
    """
    Split time series into segments and randomly permute them.
    Breaks temporal dependencies while keeping local patterns.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        max_segments: Maximum number of segments to split into
        seg_mode: "equal" for equal-sized segments, "random" for random splits
        
    Returns:
        Permuted tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = permutation(x, max_segments=4, seg_mode="equal")
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device
    ret = torch.zeros_like(x)
    
    for i in range(batch_size):
        if seg_mode == "equal":
            # Equal-sized segments
            n_segments = torch.randint(1, max_segments + 1, (1,)).item()
            segment_size = seq_len // n_segments
            segments = []
            
            for seg_idx in range(n_segments):
                start = seg_idx * segment_size
                end = start + segment_size if seg_idx < n_segments - 1 else seq_len
                segments.append(x[i, start:end, :])
            
            # Randomly permute segments
            perm = torch.randperm(n_segments, device=device)
            permuted = [segments[perm[j]] for j in range(n_segments)]
            ret[i] = torch.cat(permuted, dim=0)
            
        else:  # random
            # Random split points
            n_segments = torch.randint(2, max_segments + 1, (1,)).item()
            split_points = torch.sort(torch.randint(0, seq_len, (n_segments - 1,), device=device))[0]
            split_points = torch.cat([torch.tensor([0], device=device), split_points, torch.tensor([seq_len], device=device)])
            
            segments = []
            for j in range(len(split_points) - 1):
                segments.append(x[i, split_points[j]:split_points[j+1], :])
            
            # Randomly permute segments
            perm = torch.randperm(len(segments), device=device)
            permuted = [segments[perm[j]] for j in range(len(segments))]
            ret[i] = torch.cat(permuted, dim=0)
    
    return ret


def magnitude_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    """
    Warp magnitude using cubic spline with random knots.
    Smoothly distorts the amplitude of time series.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        sigma: Standard deviation of random knot values
        knot: Number of knots for cubic spline (including start/end)
        
    Returns:
        Warped tensor with same shape as input
        
    Note:
        Requires scipy for cubic spline interpolation. Falls back to linear
        interpolation if scipy is not available.
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = magnitude_warp(x, sigma=0.3, knot=4)
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        # Fallback to linear interpolation if scipy not available
        print("Warning: scipy not available, using linear interpolation instead of cubic spline")
        return _magnitude_warp_linear(x, sigma, knot)
    
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    # Convert to numpy for scipy interpolation
    x_np = x.cpu().numpy()
    ret_np = np.zeros_like(x_np)
    
    orig_steps = np.arange(seq_len)
    
    for i in range(batch_size):
        # Random knot points
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, n_features))
        
        # Interpolation points
        warp_steps = np.linspace(0, seq_len - 1, num=knot + 2)
        
        for dim in range(n_features):
            # Cubic spline through random warps
            cs = CubicSpline(warp_steps, random_warps[:, dim])
            # Evaluate at all time steps
            warper = cs(orig_steps)
            ret_np[i, :, dim] = x_np[i, :, dim] * warper
    
    return torch.from_numpy(ret_np).to(device)


def _magnitude_warp_linear(x: torch.Tensor, sigma: float, knot: int) -> torch.Tensor:
    """Linear interpolation fallback for magnitude_warp."""
    batch_size, seq_len, n_features = x.shape
    device = x.device
    ret = torch.zeros_like(x)
    
    for i in range(batch_size):
        # Random knot points
        random_warps = torch.randn(knot + 2, n_features, device=device) * sigma + 1.0
        
        # Linear interpolation
        warp_steps = torch.linspace(0, seq_len - 1, knot + 2, device=device)
        orig_steps = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        for dim in range(n_features):
            # Linear interpolation
            warper = torch.nn.functional.interpolate(
                random_warps[:, dim].unsqueeze(0).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze()
            ret[i, :, dim] = x[i, :, dim] * warper
    
    return ret


def time_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    """
    Warp time axis using cubic spline with random knots.
    Smoothly distorts the temporal dimension (speed up/slow down).
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        sigma: Standard deviation of random time warp
        knot: Number of knots for cubic spline (including start/end)
        
    Returns:
        Time-warped tensor with same shape as input
        
    Note:
        Requires scipy for cubic spline interpolation.
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = time_warp(x, sigma=0.3, knot=4)
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        print("Warning: scipy not available, using linear interpolation instead of cubic spline")
        return _time_warp_linear(x, sigma, knot)
    
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    # Convert to numpy for scipy interpolation
    x_np = x.cpu().numpy()
    ret_np = np.zeros_like(x_np)
    
    orig_steps = np.arange(seq_len)
    
    for i in range(batch_size):
        # Random warp with cumsum to ensure monotonicity
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        warp_steps = np.linspace(0, seq_len - 1, num=knot + 2)
        
        # Cubic spline for time warping
        cs = CubicSpline(warp_steps, warp_steps * random_warps)
        time_warp_map = cs(orig_steps)
        
        # Clip to valid range
        time_warp_map = np.clip(time_warp_map, 0, seq_len - 1)
        
        for dim in range(n_features):
            # Interpolate values at warped time points
            ret_np[i, :, dim] = np.interp(orig_steps, time_warp_map, x_np[i, :, dim])
    
    return torch.from_numpy(ret_np).to(device)


def _time_warp_linear(x: torch.Tensor, sigma: float, knot: int) -> torch.Tensor:
    """Linear interpolation fallback for time_warp."""
    batch_size, seq_len, n_features = x.shape
    device = x.device
    ret = torch.zeros_like(x)
    
    for i in range(batch_size):
        # Random time warp factors
        random_warps = torch.randn(knot + 2, device=device) * sigma + 1.0
        warp_steps = torch.linspace(0, seq_len - 1, knot + 2, device=device)
        
        # Simple linear time warp
        time_warp_map = torch.linspace(0, seq_len - 1, seq_len, device=device)
        time_warp_map = time_warp_map * (1.0 + torch.randn(1, device=device).item() * sigma * 0.1)
        time_warp_map = torch.clamp(time_warp_map, 0, seq_len - 1)
        
        for dim in range(n_features):
            # Use grid_sample for interpolation
            grid = (time_warp_map / (seq_len - 1) * 2 - 1).view(1, 1, -1, 1)  # normalize to [-1, 1]
            values = x[i, :, dim].view(1, 1, seq_len, 1)
            warped = torch.nn.functional.grid_sample(
                values, grid, mode='bilinear', align_corners=True, padding_mode='border'
            )
            ret[i, :, dim] = warped.squeeze()
    
    return ret


def window_slice(x: torch.Tensor, reduce_ratio: float = 0.9) -> torch.Tensor:
    """
    Slice a window from time series and interpolate to original length.
    Creates variations by zooming into different parts of the series.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        reduce_ratio: Ratio of window size to sequence length (0 < ratio < 1)
        
    Returns:
        Sliced and interpolated tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = window_slice(x, reduce_ratio=0.8)
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    # Window size
    target_len = int(np.ceil(reduce_ratio * seq_len))
    
    if target_len >= seq_len:
        return x
    
    ret = torch.zeros_like(x)
    
    for i in range(batch_size):
        # Random start position
        start = torch.randint(0, seq_len - target_len + 1, (1,), device=device).item()
        end = start + target_len
        
        # Extract window
        window = x[i, start:end, :]  # (target_len, n_features)
        
        # Interpolate back to original length
        # Reshape for interpolate: (1, n_features, target_len) -> (1, n_features, seq_len)
        window_T = window.T.unsqueeze(0)  # (1, n_features, target_len)
        interpolated = torch.nn.functional.interpolate(
            window_T, 
            size=seq_len, 
            mode='linear', 
            align_corners=True
        )
        ret[i] = interpolated.squeeze(0).T  # (seq_len, n_features)
    
    return ret


def window_warp(x: torch.Tensor, window_ratio: float = 0.1, scales: tuple = (0.5, 2.0)) -> torch.Tensor:
    """
    Randomly warp windows within time series.
    Speeds up or slows down specific temporal windows.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        window_ratio: Ratio of window size to sequence length
        scales: Tuple of (min_scale, max_scale) for window warping
        
    Returns:
        Warped tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = window_warp(x, window_ratio=0.2, scales=(0.5, 2.0))
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    # Convert to numpy for easier interpolation
    x_np = x.cpu().numpy()
    ret_np = np.zeros_like(x_np)
    
    warp_size = int(np.ceil(window_ratio * seq_len))
    window_steps = np.arange(warp_size)
    
    for i in range(batch_size):
        # Random window position
        start = np.random.randint(0, seq_len - warp_size)
        end = start + warp_size
        
        # Random warp scale
        warp_scale = np.random.uniform(scales[0], scales[1])
        
        for dim in range(n_features):
            # Extract segments
            start_seg = x_np[i, :start, dim]
            window_seg = x_np[i, start:end, dim]
            end_seg = x_np[i, end:, dim]
            
            # Warp the window segment
            window_warped = np.interp(
                np.linspace(0, warp_size - 1, num=int(warp_size * warp_scale)),
                window_steps,
                window_seg
            )
            
            # Concatenate and interpolate back to original length
            warped = np.concatenate((start_seg, window_warped, end_seg))
            ret_np[i, :, dim] = np.interp(
                np.arange(seq_len),
                np.linspace(0, seq_len - 1, num=warped.size),
                warped
            )
    
    return torch.from_numpy(ret_np).to(device)


# ==============================================================================
# High-level augmentation functions
# ==============================================================================

def apply_augmentations(
    x: torch.Tensor,
    augmentation_list: list[str],
    **kwargs
) -> torch.Tensor:
    """
    Apply multiple augmentations sequentially.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        augmentation_list: List of augmentation names to apply
        **kwargs: Additional parameters for specific augmentations
            - jitter_sigma: float = 0.03
            - scaling_sigma: float = 0.1
            - mag_warp_sigma: float = 0.2
            - mag_warp_knot: int = 4
            - time_warp_sigma: float = 0.2
            - time_warp_knot: int = 4
            - permutation_segments: int = 5
            - permutation_mode: str = "equal"
            - window_slice_ratio: float = 0.9
            - window_warp_ratio: float = 0.1
            - window_warp_scales: tuple = (0.5, 2.0)
            
    Returns:
        Augmented tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = apply_augmentations(
        ...     x, 
        ...     ["jitter", "scaling", "permutation"],
        ...     jitter_sigma=0.05,
        ...     scaling_sigma=0.15
        ... )
    """
    x_aug = x.clone()
    
    for aug_name in augmentation_list:
        if aug_name == "jitter":
            sigma = kwargs.get("jitter_sigma", 0.03)
            x_aug = jitter(x_aug, sigma=sigma)
            
        elif aug_name == "scaling":
            sigma = kwargs.get("scaling_sigma", 0.1)
            x_aug = scaling(x_aug, sigma=sigma)
            
        elif aug_name == "rotation":
            x_aug = rotation(x_aug)
            
        elif aug_name == "permutation":
            max_segments = kwargs.get("permutation_segments", 5)
            seg_mode = kwargs.get("permutation_mode", "equal")
            x_aug = permutation(x_aug, max_segments=max_segments, seg_mode=seg_mode)
            
        elif aug_name == "magnitude_warp" or aug_name == "magwarp":
            sigma = kwargs.get("mag_warp_sigma", 0.2)
            knot = kwargs.get("mag_warp_knot", 4)
            x_aug = magnitude_warp(x_aug, sigma=sigma, knot=knot)
            
        elif aug_name == "time_warp" or aug_name == "timewarp":
            sigma = kwargs.get("time_warp_sigma", 0.2)
            knot = kwargs.get("time_warp_knot", 4)
            x_aug = time_warp(x_aug, sigma=sigma, knot=knot)
            
        elif aug_name == "window_slice" or aug_name == "windowslice":
            ratio = kwargs.get("window_slice_ratio", 0.9)
            x_aug = window_slice(x_aug, reduce_ratio=ratio)
            
        elif aug_name == "window_warp" or aug_name == "windowwarp":
            ratio = kwargs.get("window_warp_ratio", 0.1)
            scales = kwargs.get("window_warp_scales", (0.5, 2.0))
            x_aug = window_warp(x_aug, window_ratio=ratio, scales=scales)
            
        else:
            print(f"Warning: Unknown augmentation '{aug_name}', skipping")
    
    return x_aug


def random_augmentation(
    x: torch.Tensor,
    num_augmentations: int = 2,
    available_augs: Optional[list[str]] = None
) -> torch.Tensor:
    """
    Apply random augmentations from available list.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        num_augmentations: Number of augmentations to apply
        available_augs: List of available augmentation names. If None, uses all basic augs.
        
    Returns:
        Augmented tensor with same shape as input
        
    Example:
        >>> x = torch.randn(32, 96, 7)
        >>> x_aug = random_augmentation(x, num_augmentations=3)
    """
    if available_augs is None:
        available_augs = ["jitter", "scaling", "rotation", "permutation", "window_slice"]
    
    # Randomly select augmentations
    selected_augs = np.random.choice(available_augs, size=min(num_augmentations, len(available_augs)), replace=False).tolist()
    
    return apply_augmentations(x, selected_augs)
