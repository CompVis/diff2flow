"""
Adapted from https://github.com/madebyollin/taesd
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TinyAutoencoderKL(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels is None:
            latent_channels = self.guess_latent_channels(str(encoder_path))
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def encode(self, x):
        """
        Args:
            x: torch.Tensor, shape (b, 3, h, w) in [-1, 1]
        """
        # scale to [0, 1]
        x = x.div(2).add(0.5)
        return self.encoder(x)
    
    @torch.no_grad()
    def decode(self, z):
        """
        Args:
            z: torch.Tensor, shape (b, latent_channels, h, w)
        """
        # scale to [-1, 1]
        return self.decoder(z).mul(2).sub(1)

    def guess_latent_channels(self, encoder_path):
        """guess latent channel count based on encoder filename"""
        if "taef1" in encoder_path:
            return 16
        if "taesd3" in encoder_path:
            return 16
        return 4

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TinyAutoencoderKL.latent_magnitude).add(TinyAutoencoderKL.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TinyAutoencoderKL.latent_shift).mul(2 * TinyAutoencoderKL.latent_magnitude)


if __name__ == "__main__":
    inp = torch.randn((1, 3, 256, 256))
    model = TinyAutoencoderKL(encoder_path=None, decoder_path=None)

    z = model.encode(inp)
    out = model.decode(z)
    print(f"{'Input':<10}: {inp.shape}")
    print(f"{'Encoded':<10}: {z.shape}")
    print(f"{'Output':<10}: {out.shape}")
