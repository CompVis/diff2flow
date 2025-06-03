import torch
import einops
import numpy as np
import torch.nn as nn
from functools import partial

from diff2flow.helpers import instantiate_from_config
from diff2flow.ddpm import GaussianDiffusion
from diff2flow.ddim import DDIMSampler


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def shifted_cosine_log_snr(t, im_size: int, ref_size: int = 64):
    return cosine_log_snr(t) + 2 * np.log(ref_size / im_size)


def log_snr_to_alpha_bar(t):
    return sigmoid(cos_log_snr(t))


def shifted_cosine_alpha_bar(t, im_size: int, ref_size: int = 64):
    return sigmoid(shifted_cosine_log_snr(t, im_size, ref_size))


class ForwardDiffusion(nn.Module):
    def __init__(self,
                 im_size: int = 64,
                 n_diffusion_timesteps: int = 1000):
        super().__init__()
        self.n_diffusion_timesteps = n_diffusion_timesteps
        cos_alpha_bar_t = shifted_cosine_alpha_bar(
            np.linspace(0, 1, n_diffusion_timesteps),
            im_size=im_size
        ).astype(np.float32)
        self.register_buffer("alpha_bar_t", torch.from_numpy(cos_alpha_bar_t))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps. In other
        words sample from q(x_t | x_0).

        Args:
            x_start: The initial data batch.
            t: The diffusion time-step (must be a single t).
            noise: If specified, the noise to use for the diffusion.
        Returns:
            A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_bar_t = self.alpha_bar_t[t]

        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
        

""" Diffusion Wrapper adapted to FlowModel """


class DiffusionFlow(nn.Module):
    def __init__(
            self,
            net_cfg: dict,
            timesteps: int = 1000,
            beta_schedule: str = 'linear',
            loss_type: str = 'l2',
            parameterization: str = 'v',
            linear_start: float = 0.00085,
            linear_end: float = 0.0120,
            ddim_steps: int = 10,           # 10 timesteps default for FlowModel
    ):
        super().__init__()
        self.net = instantiate_from_config(net_cfg)

        self.diffusion_cfg = dict(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            zero_terminal_snr=False,
            loss_type=loss_type,
            parameterization=parameterization,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
            original_elbo_weight=0.,
            v_posterior=0.,
            l_simple_weight=1.0
        )
        self.diffusion = GaussianDiffusion(**self.diffusion_cfg)

        self.ddim_steps = ddim_steps
        self.ddim_sampler = DDIMSampler(self.diffusion)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.net(x, t, **kwargs)

    def training_losses(self, x1: torch.Tensor, x0: torch.Tensor = None, **cond_kwargs):
        loss, _ = self.diffusion.training_losses(
            model=self.net,
            x_start=x1,
            model_kwargs=cond_kwargs,
            x_noise=x0,
        )
        return loss
    
    def generate(self, x: torch.Tensor, sample_kwargs=None, reverse=False, return_intermediates=False, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            sample_kwargs: dict, additional sampling arguments for the solver
                progress: bool, whether to show a progress bar
                clip_denoised: bool, whether to clip the denoised images to [-1, 1]
                use_ddpm: bool, whether to use DDPM sampling instead of DDIM
                intermediate_key: str, key to use for intermediate outputs
                    (DDIM: 'x_inter', 'pred_x0' | DDPM: 'sample' or 'pred_xstart')
                intermediate_freq: int, frequency of intermediate outputs
                __ DDIM only __:
                    num_steps: int, number of DDIM steps to take
                    eta: float, noise level for DDIM
                    temperature: float, temperature for DDIM
                    noise_dropout: float, dropout rate for DDIM
                    cfg_scale: float, scale factor for Classifier-free guidance
                    uc_cond: torch.Tensor, unconditional conditioning information
                    cond_key: str, key to use for conditioning information
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: if true, return the intermediate samples
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        if reverse:
            raise NotImplementedError("[DiffusionFlow] Reverse sampling not yet supported")
        
        sample_kwargs = sample_kwargs or {}
        
        # DDPM sampling
        if sample_kwargs.get("use_ddpm", False):
            # include CFG
            forward_fn = partial(
                forward_with_cfg,
                model       = self.net,
                cfg_scale   = sample_kwargs.get("cfg_scale", 1.),
                uc_cond     = sample_kwargs.get("uc_cond", None),
                cond_key    = sample_kwargs.get("cond_key", "y"),
            )
            out, intermediates = self.diffusion.p_sample_loop(
                # model                   = self.net,           # without CFG
                model                   = forward_fn,
                noise                   = x,
                model_kwargs            = kwargs,
                progress                = sample_kwargs.get("progress", False),
                clip_denoised           = sample_kwargs.get("clip_denoised", False),
                return_intermediates    = True,
                intermediate_freq       = sample_kwargs.get("intermediate_freq", (100 if return_intermediates else 1000)),
                pbar_desc               = sample_kwargs.get("pbar_desc", "DDPM Sampling"),
                intermediate_key        = sample_kwargs.get("intermediate_key", "sample"),
            )

        # DDIM sampling
        else:
            out, intermediates = self.ddim_sampler.sample(
                model                   = self.net,
                noise                   = x,
                model_kwargs            = kwargs,
                ddim_steps              = sample_kwargs.get("num_steps", self.ddim_steps),
                eta                     = sample_kwargs.get("eta", 0.),
                progress                = sample_kwargs.get("progress", False),
                temperature             = sample_kwargs.get("temperature", 1.),
                noise_dropout           = sample_kwargs.get("noise_dropout", 0.),
                log_every_t             = sample_kwargs.get("intermediate_freq", (10 if return_intermediates else 1000)),
                clip_denoised           = sample_kwargs.get("clip_denoised", False),
                cfg_scale               = sample_kwargs.get("cfg_scale", 1.),
                uc_cond                 = sample_kwargs.get("uc_cond", None),
                cond_key                = sample_kwargs.get("cond_key", "y"),
            )
            key = sample_kwargs.get("intermediate_key", "x_inter")
            intermediates = intermediates[key]

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return out


def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    if cfg_scale == 1.0:                                # without CFG
        model_output = model(x, t, **model_kwargs)

    else:                                               # with CFG
        assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        assert uc_cond is not None, "Unconditional condition not provided for CFG"
        kwargs = model_kwargs.copy()
        c = kwargs[cond_key]
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if uc_cond.shape[0] == 1:
            uc_cond = einops.repeat(uc_cond, '1 ... -> bs ...', bs=x.shape[0])
        c_in = torch.cat([uc_cond, c])
        kwargs[cond_key] = c_in
        model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)

    return model_output


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = ForwardDiffusion(im_size=64)

    plt.plot(d.alpha_bar_t)
    plt.plot(1 - d.alpha_bar_t)
    plt.savefig("cosine.png")
    plt.close()

    im = torch.ones((1, 3, 128, 128))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        xt = d.q_sample(im, i * 100)
        print(f"t={i*100}: {d.alpha_bar_t[i*100]}")
        ax.imshow(xt[0].permute(1, 2, 0).detach().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("diffusion.png")
    plt.close()
