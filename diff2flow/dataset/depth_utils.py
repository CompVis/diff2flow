import torch
import einops
import warnings
import numpy as np
import matplotlib.pyplot as plt


def exists(v):
    return v is not None


def get_scale_and_shift(img1, img2, mask=None):
    """
        Returns the scale and shift between two images
        More precisely, we find (b1, b2) such that img1 = b1 + b2*img2
        
        We model this problem as a least-squares problem:
        (b1*, b2*) = argmin_(b1, b2) ||img1 - b1 + b2*img2||_(L2)   
        
    """
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # we might get some nans and infs here, exclude them first
    valid1 = torch.logical_and(~torch.isinf(img1_flat), ~torch.isinf(img2_flat))
    valid2 = torch.logical_and(~torch.isnan(img1_flat), ~torch.isnan(img2_flat))
    valid = torch.logical_and(valid1, valid2)

    # apply mask
    if mask is not None:
        mask_flat = mask.flatten()
        valid = torch.logical_and(valid, mask_flat)

    img1_flat = img1_flat[valid]
    img2_flat = img2_flat[valid]
    
    ones = torch.ones_like(img1_flat)
    X = torch.cat((ones[None, ...], img2_flat[None, ...]), dim=0).T
    
    # compute analytical solution
    b_opt = (X.T@X).float().inverse()@X.T@img1_flat     # works with bf16-mixed training
    shift, scale = b_opt
    return shift, scale


def get_batch_scale_and_shift(img1, img2, mask=None):
    """
        Returns the scale and shift between two images
        More precisely, we find (b1, b2) such that img1 = b1 + b2*img2
        
        We model this problem as a least-squares problem:
        (b1*, b2*) = argmin_(b1, b2) ||img1 - b1 + b2*img2||_(L2)   
        
    """
    assert len(img1.shape) == 4, "img1 must be of shape (batch_size, channels, height, width)"
    shifts, scales = [], []
    for i in range(img1.shape[0]):
        shift, scale = get_scale_and_shift(img1[i], img2[i], None if mask is None else mask[i])
        shifts.append(shift)
        scales.append(scale)
    shifts = torch.stack(shifts)
    scales = torch.stack(scales)
    return shifts, scales


def apply_scale_and_shift(pred, gt, mask=None):
    """
    Returns a scale-and-shift version of pred according to gt

    Args:
        pred: predicted image (b, c, h, w)
        gt  : ground truth image (b, c, h, w)
    """
    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    if mask is not None:
        assert pred.shape == mask.shape, "pred and mask must have the same shape"
    shifts, scales = get_batch_scale_and_shift(gt, pred, mask=mask)
    shifts = shifts[:, None, None, None]
    scales = scales[:, None, None, None]
    pred_scaled = shifts + scales*pred
    return pred_scaled


def abs_rel_error(pred, target, valid_mask=None):
    if pred.shape[1] == 3:
        pred = pred.mean(dim=1, keepdim=True)
    if target.shape[1] == 3:
        target = target.mean(dim=1, keepdim=True)
    if valid_mask is not None:
        pred = pred[valid_mask]
        target = target[valid_mask]
    return torch.mean(torch.abs(pred - target) / target)


def delta1_accuracy(pred, target):
    if pred.shape[1] == 3:
        pred = pred.mean(dim=1, keepdim=True)
    if target.shape[1] == 3:
        target = target.mean(dim=1, keepdim=True)
    return torch.mean((torch.max(pred / target, target / pred) < 1.25).float())


def percentile_per_sample(x, percentile):
    return np.percentile(x, percentile, axis=[*range(1, x.ndim)])


def pad_vector_like_x(v, x):  
    """  
    Function to reshape the vector by the number of dimensions 
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).  
    Args:
        x : Tensor, shape (bs, *dim)
        v : FloatTensor, shape (bs)  
    Returns:
        vec : Tensor, shape (bs, number of x dimensions)
    """
    if isinstance(v, float):  
        return v  
    return v.reshape(-1, *([1] * (x.ndim - 1)))


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


def colorize_depth_map(
        depth,
        vmin=None,
        vmax=None,
        percentiles=False,
        cmap="Spectral",
        invalid_mask=None,
        invalid_color=(0, 0, 0),
        inverse=False
    ):
    """
    Colorize a depth map using a matplotlib colormap.
    
    Args:
        depth: Depth tensor of shape (b, 1, h, w) or (b, h, w) with
            planar depth values ranging from 0 to inf.
        vmin: Minimum depth value to use for scaling the colormap. Can
            also be a percentile value if percentiles is True. If None,
            values in the batch are not min-clipped.
        vmax: Maximum depth value to use for scaling the colormap. Can
            also be a percentile value if percentiles is True. If None,
            values in the batch are not max-clipped.
        percentiles: If True, vmin and vmax are interpreted as percentiles
            of the depth values in the batch (per sample!).
        cmap: Name of the matplotlib colormap to use.
        invalid_mask: Boolean mask of shape (b, h, w) that is True where
            the depth values are invalid.
        invalid_color: RGB color to use for invalid depth values.
        inverse: If True, the depth values are inverted before colorization.
    Returns:
        Colorized depth map of shape (b, h, w, 3) with RGB values [0, 255].
    """
    if len(depth.shape) == 4:
        assert depth.shape[1] == 1, "Depth must have 1 channel."
        depth = depth.squeeze(1)
    assert len(depth.shape) == 3, "Depth must have shape (b, h, w) or (b, 1, h, w)."
    # assert depth.min() >= 0, "Depth must be non-negative."

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    
    # clip values with vmin and vmax
    if vmin is not None and percentiles:
        assert 0 <= vmin < 100, "vmin must be in [0, 100] if using percentiles"
        vmin = percentile_per_sample(depth, vmin)
        vmin = pad_vector_like_x(vmin, depth)
    if vmax is not None and percentiles:
        assert 0 < vmax <= 100, "vmax must be in [0, 100] if using percentiles"
        vmax = percentile_per_sample(depth, vmax)
        vmax = pad_vector_like_x(vmax, depth)
    if exists(vmin) or exists(vmax):
        # clip values between vmin and vmax
        depth = np.clip(depth, vmin, vmax)

    # take inverse of depth
    if inverse:
        depth = 1.0 / depth

    # normalize to [0, 1]
    depth = per_sample_min_max_normalization(depth)
    
    # apply colormap
    cmapper = plt.get_cmap(cmap)
    depth = cmapper(depth, bytes=True)[..., :3]         # (b, h, w, 3)

    if invalid_mask is not None:
        depth[invalid_mask] = invalid_color

    return depth
