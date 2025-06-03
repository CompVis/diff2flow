import torch
import einops
import numpy as np
from PIL import Image


def ims_to_grid(ims, stack="row", split=4, channel_last=False):
    """
    Args:
        ims: Tensor of shape (b, c, h, w)
        stack: "row" or "col"
        split: If 'row' stack by rows, if 'col' stack by columns.
    Returns:
        Tensor of shape (h, w, c)
    """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    from_ = 'h w c' if channel_last else 'c h w'
    if split is not None and ims.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        ims = einops.rearrange(ims, f"(b1 b2) {from_} -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        ims = einops.rearrange(ims, f"b {from_} -> " + to)
    return ims


def tensor2im(tensor, denormalize_zero_one=False):
    """
    Args:
        tensor: Tensor of shape (..., 3, h, w) in range [-1, 1] or [0, 1]
        denormalize_zero_one: If True, denormalizes image from range [0, 1] otherwise
            from [-1, 1] to [0, 255]
    Returns:
        Numpy array of shape (h, w, 3) in range [0, 255] if tensor shape is (3, h, w)
            or (1, 3, h, w). Otherwise, returns array of shape (..., h, w, 3).
    """
    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    im = einops.rearrange(tensor, '... c h w -> ... h w c')
    if denormalize_zero_one:
        im = im * 255.
    else:
        im = (im + 1.) * 127.5
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im


class ImageVisualizer:
    def __call__(self, *images) -> Image:
        # images: list of tensors that is not normalized in shape of [b c h w]
        images = [i for i in images if i is not None]       # hack to exclude None (e.g. x0 is noise)
        images = torch.stack(images) #  [n b c h w]
        images = einops.rearrange(images, 'n b c h w -> (b h) (n w) c')
        images = images / 2 + 0.5
        images = images.cpu().numpy()
        images = np.clip(images, 0, 1)
        images = (images * 255).astype(np.uint8)
        images = Image.fromarray(images)
        return images


class T2IVisualizer:
    def __init__(self, show_x1=False):
        self.show_x1 = show_x1
    
    def __call__(self, x0=None, x1=None, x1_pred=None):
        if self.show_x1:
            images = torch.cat([x1, x1_pred], dim=-1)       # [b c h (2w)]
            images = tensor2im(images)     # (b h w c) in [0, 255]
            images = einops.rearrange(images, 'b h w c -> (b h) w c')
            images = Image.fromarray(images)
        else:
            # ignore x0 and x1, only visualize x1_pred
            images = tensor2im(x1_pred)     # (b h w c) in [0, 255]
            images = ims_to_grid(images, stack="row", split=2, channel_last=True)   # (h w c)
            images = Image.fromarray(images)
        return images


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


class ImageDepthVisualizer:
    def __call__(self, img, depth, depth_pred) -> Image:
        # min-max normalize depth per sample
        depth = per_sample_min_max_normalization(depth)
        depth_pred = per_sample_min_max_normalization(depth_pred)
        depth = depth.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        depth_pred = depth_pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        img = img / 2 + 0.5
        # => all three should be in shape of [b 3 h w] and [0, 1]
        
        # concatenate along width
        out = torch.cat([img, depth, depth_pred], dim=3)
        out = einops.rearrange(out, "b c h w -> (b h) w c")
        out = (out * 255).clip(0, 255).cpu().numpy().astype('uint8')
        out = Image.fromarray(out)
        return out
