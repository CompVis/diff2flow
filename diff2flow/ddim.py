""" SAMPLING ONLY.
Adapted from https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py
"""
import os
import sys
import torch
import einops
import warnings
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try: from .ddpm import GaussianDiffusion
except ImportError: from ddpm import GaussianDiffusion


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        assert c > 1, "Issue, not solved!"
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
        
    return sigmas, alphas, alphas_prev


class DDIMSampler:
    def __init__(self, ddpm: GaussianDiffusion, schedule="linear"):
        super().__init__()
        self.ddpm = ddpm
        self.ddpm_num_timesteps = ddpm.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, device, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.device = device
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.ddpm.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.ddpm.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.ddpm.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        ddim_steps,
        eta=0.,
        model_kwargs=None,
        progress=True,
        temperature=1.,
        noise_dropout=0.,
        clip_denoised=False,
        log_every_t=100,
        cfg_scale=1.,
        uc_cond=None,
        cond_key='y'
    ):
        bs, dev = noise.shape[0], noise.device

        self.make_schedule(ddim_num_steps=ddim_steps, device=dev, ddim_eta=eta, verbose=False)

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=not progress)

        img = noise
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=dev, dtype=torch.long)

            outs = self.p_sample_ddim(
                model=model,
                x=img,
                t=ts,
                index=index,
                temperature=temperature,
                noise_dropout=noise_dropout,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
                uc_cond=uc_cond,
                cond_key=cond_key
            )
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def p_sample_ddim(
            self,
            model,
            x,
            t,
            index,
            use_original_steps=False,
            temperature=1.,
            noise_dropout=0.,
            model_kwargs=None,
            clip_denoised=False,
            cfg_scale=1.,
            uc_cond=None,
            cond_key='y'
        ):
        model_kwargs = model_kwargs or {}

        # model prediction
        if cfg_scale == 1.:         # without CFG
            model_output = model(x, t, **model_kwargs)
        else:                       # with CFG
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

        if self.ddpm.parameterization == "v":
            e_t = self.ddpm.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        # inference schedule
        alphas = self.ddpm.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.ddpm.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddpm.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        bs, dev = x.shape[0], x.device
        a_t = torch.full((bs, 1, 1, 1), alphas[index], device=dev)
        a_prev = torch.full((bs, 1, 1, 1), alphas_prev[index], device=dev)
        sigma_t = torch.full((bs, 1, 1, 1), sigmas[index], device=dev)
        sqrt_one_minus_at = torch.full((bs, 1, 1, 1), sqrt_one_minus_alphas[index], device=dev)

        # current prediction for x_0
        if self.ddpm.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.ddpm.predict_start_from_z_and_v(x, t, model_output)

        if clip_denoised:
            pred_x0 = pred_x0.clamp_(-1., 1.)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(x) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0


    """ Inversion """
    
    @torch.no_grad()
    def encode(
        self,
        model,
        x0,
        t_enc=None,
        ddim_steps=100,
        use_original_steps=True,
        n_intermediates=0,
        model_kwargs=None,
        progress=True
    ):
        assert self.ddpm.parameterization == "eps", "Only works with eps parameterization"
        if not use_original_steps:
            warnings.warn('Using DDIM for encoding is not recommended, as it is not fully debugged.')
        
        bs, dev = x0.shape[0], x0.device
        model_kwargs = model_kwargs or {}
        return_intermediates = n_intermediates > 0

        self.make_schedule(ddim_num_steps=ddim_steps, device=dev, ddim_eta=0.0, verbose=False)

        # steps
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]
        if t_enc is None:
            t_enc = num_reference_steps
        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.ddpm.alphas_cumprod[:num_steps]
            alphas = self.ddpm.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding x0', disable=not progress):
            t = torch.full((bs,), i, device=dev, dtype=torch.long)
            
            noise_pred = model(x_next, t, **model_kwargs)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (num_steps // n_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def decode(
        self,
        model,
        x_latent,
        t_start=None,
        ddim_steps=100,
        use_original_steps=True,
        model_kwargs=None,
        progress=True,
        clip_denoised=False
    ):
        bs, dev = x_latent.shape[0], x_latent.device
        model_kwargs = model_kwargs or {}

        self.make_schedule(ddim_num_steps=ddim_steps, device=dev, ddim_eta=0.0, verbose=False)

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        if t_start is not None:
            timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps, disable=not progress)
        
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=dev, dtype=torch.long)

            x_dec, _ = self.p_sample_ddim(
                model=model,
                x=x_dec,
                t=ts,
                index=index,
                model_kwargs=model_kwargs,
                use_original_steps=use_original_steps,
                clip_denoised=clip_denoised
            )

        return x_dec
