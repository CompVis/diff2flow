import os
import torch
import open_clip
import numpy as np
from torch import nn
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTokenizer, CLIPTextModel


# Models in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
class ClipImageEmbedder(nn.Module):
    def __init__(
            self,
            model="ViT-L/14",
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=True,
            ucg_rate=0.
    ):
        super().__init__()
        from clip import load as load_clip
        self.model, _ = load_clip(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

        self.init_uncond()

    def init_uncond(self, path="nulltext.npy"):
        try:
            assert os.path.exists(path), f"Uncond file {path} not found."
            print(f"Loading uncond from {path}")
            uncond = torch.from_numpy(np.load(path))
            self.register_buffer("uncond", uncond)
        except:
            self.uncond = None

    def preprocess(self, x):
        # resize to 224, normalize to [0,1] and re-normalize according to clip
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        assert x.min() >= -1. and x.max() <= 1.
        x = (x + 1.) / 2.
        x = (x - self.mean.reshape(1,-1,1,1)) / self.std.reshape(1,-1,1,1)
        return x

    def forward(self, x, no_dropout=False):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0. and not no_dropout:
            out = torch.bernoulli((1. - self.ucg_rate) * torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out.unsqueeze(1)
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, device="cuda"):
        if self.uncond is None:
            raise ValueError("Unconditional conditioning not initialized.")
        return self.uncond.to(device)


class DummyOpenCLIPTextEmbedder(nn.Module):
    def __init__(
            self,
            nulltext_path="nulltext.npy",
    ):
        super().__init__()
        self.arch = "ViT-H-14"
        self.version = "laion2b_s32b_b79k"
        self.layer = "penultimate"

        # find correct path for nulltext embedding
        if isinstance(nulltext_path, list) or isinstance(nulltext_path, ListConfig):
            for path in nulltext_path:
                if os.path.exists(path):
                    nulltext_path = path
                    break
            else:
                raise FileNotFoundError("Could not find a valid nulltext path.")

        cond = np.load(nulltext_path)
        self.uncond = torch.from_numpy(cond).squeeze(0)

    def forward(self, x, *args, **kwargs):
        bs = x.shape[0]
        device = x.device
        if self.uncond.device is not device:
            self.uncond = self.uncond.to(device)
        
        return self.uncond[None, :].repeat(bs, 1, 1)


class FrozenOpenCLIPImageEmbedder(nn.Module):
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="pooled", antialias=True, ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'),
                                                            pretrained=version, )
        del model.transformer
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # resize to 224, normalize to [0,1] and re-normalize according to clip
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        assert x.min() >= -1. and x.max() <= 1.
        x = (x + 1.) / 2.
        x = (x - self.mean.reshape(1,-1,1,1)) / self.std.reshape(1,-1,1,1)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli((1. - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z.unsqueeze(1)

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ["last", "penultimate"]
    def __init__(
            self,
            arch="ViT-H-14",
            version="laion2b_s32b_b79k",
            max_length=77,
            freeze=True,
            layer="penultimate"
        ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        
        self.uncond = None

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        dev = next(self.parameters()).device
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(dev))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, device="cuda"):
        """
        Returns:
            torch.Tensor: Unconditional conditioning information for text
                of shape (1, max_length, d_model), e.g. (1, 77, 1024)
        """
        if self.uncond is None:
            self.uncond = self.encode("")
        return self.uncond.to(device)
    
class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.uncond = None

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        dev = next(self.parameters()).device
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(dev)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, device="cuda"):
        """
        Returns:
            torch.Tensor: Unconditional conditioning information for text
                of shape (1, max_length, d_model), e.g. (1, 77, 1024)
        """
        if self.uncond is None:
            self.uncond = self.encode("")
        return self.uncond.to(device)


if __name__ == "__main__":
    # Test CLIP embedder
    clip = FrozenCLIPEmbedder()
    clip = clip.to('cuda:0')
    text = "A photo of a cat."
    z = clip(text)
    print(z.shape)      # (1, 77, 1024)
    uc = clip.get_unconditional_conditioning()
    print(uc.shape)     # (77, 1024)
