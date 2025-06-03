import torch
import torch.nn as nn
import torchvision.transforms as T


class CenterCropResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        h, w = x.shape[-2:]
        min_size = min(h, w)
        crop = T.functional.center_crop(x, min_size)
        x = T.functional.resize(crop, self.size, antialias=True)
        return x


class RescaleDiffusersLatent:
    def __init__(self, divisor, exclude_keys=None):
        self.divisor = divisor
        self.exclude_keys = exclude_keys or []
    
    def __call__(self, sample):
        for k, v in sample.items():
            if k.endswith("_latent") and k not in self.exclude_keys:
                latent = sample[k]
                sample[k] = latent / self.divisor
        return sample


class AddNoiseLatent:
    def __init__(self, key, resolution: int = 512, latent_dim: int = 4):
        self.key = key
        assert resolution % 8 == 0, "Resolution must be divisible by 8"
        self.shape = (latent_dim, resolution // 8, resolution // 8)

    def __call__(self, sample):
        assert self.key not in sample, f"Key {self.key} already exists in sample"
        noise = torch.randn(self.shape)
        sample[self.key] = noise
        return sample
