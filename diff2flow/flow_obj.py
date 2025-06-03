import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import numpy as np

from diff2flow.flow import FlowModel
from diff2flow.flow import forward_with_cfg

from diff2flow.utils.diffusion_utils import make_beta_schedule
from diff2flow.utils.diffusion_utils import enforce_zero_terminal_snr
from diff2flow.utils.diffusion_utils import extract_and_interpolate_into_tensor as extract_into_tensor


""" Flow Model """


class FlowModelObj(FlowModel):
    def __init__(
        self,
        enforce_zero_snr: bool = True,
        diffusion_parameterization: str = 'v',
        diffusion_schedule: str = 'linear',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.register_sdv2_schedule(diffusion_schedule, enforce_zero_snr)
        assert diffusion_parameterization in ['v', 'eps'], 'Diffusion parameterization has to be either v or eps'
        self.diffusion_parameterization = diffusion_parameterization
        self.diffusion_schedule = diffusion_schedule

    def ode_fn(self, t, x, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t, **kwargs)
        return _pred
    
    def register_sdv2_schedule(self, diffusion_schedule, enforce_zero_snr=True):
        # SDV2 schedule
        linear_start = 0.00085
        linear_end = 0.0120

        betas = make_beta_schedule(
            diffusion_schedule,
            n_timestep=1000,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        if enforce_zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_full = np.append(1., alphas_cumprod)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_full', to_torch(alphas_cumprod_full))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_alphas_cumprod_full', to_torch(np.sqrt(alphas_cumprod_full)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_full', to_torch(np.sqrt(1. - alphas_cumprod_full)))

        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.register_buffer('rectified_alphas_cumprod_full', self.sqrt_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))
        self.register_buffer('rectified_sqrt_alphas_cumprod_full', self.sqrt_one_minus_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))

    def sample_vt(self, fm_x, fm_t, **kwargs):
        """
        Sample the v-parameterized vector field at time t
        """
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        # print(fm_t, dm_t)
        dm_x = self.convert_fm_xt_to_dm_xt(fm_x, fm_t)
        # vt = self.net(dm_x, dm_t, **kwargs)
        vt = forward_with_cfg(dm_x, dm_t, self.net, **kwargs)
        
        # TODO: ugly fix for nan values!!!
        if torch.isnan(vt).any():
            vt[torch.isnan(vt)] = 0
        
        # vt = self.forward(x=dm_x, t=dm_t, **kwargs)
        if self.diffusion_parameterization == 'v':
            vector_field = self.get_vector_field_from_v(vt, dm_x, dm_t)
        elif self.diffusion_parameterization == 'eps':
            vector_field = self.get_vector_field_from_eps(vt, dm_x, dm_t)
        return vector_field

    def convert_fm_t_to_dm_t(self, t):
        """
        Convert the continuous time t in [0,1] to discrete time t [0, 1000]
        # TODO: Make it compatible with zero-terminal SNR
        """
        rectified_alphas_cumprod_full = self.rectified_alphas_cumprod_full.clone().to(t.device)
        # reverse the rectified_alphas_cumprod_full for searchsorted
        rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])
        right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True)
        left_index = right_index - 1
        right_value = rectified_alphas_cumprod_full[right_index]
        left_value = rectified_alphas_cumprod_full[left_index]
        dm_t = left_index + (t - left_value) / (right_value - left_value)
        # now reverse back the dm_t
        dm_t = self.num_timesteps - dm_t
        return dm_t
    
    def convert_fm_xt_to_dm_xt(self, fm_xt, fm_t):
        """
        Convert fm trajectory to dm trajectory using the fm t
        We use linear scaling here
        """
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        # do lienar interpolation here
        dm_t_left_index = torch.floor(dm_t)
        dm_t_right_index = torch.ceil(dm_t)
        dm_t_left_value = scale[dm_t_left_index.long()]
        dm_t_right_value = scale[dm_t_right_index.long()]

        scale_t = dm_t_left_value + (dm_t - dm_t_left_index) * (dm_t_right_value - dm_t_left_value)
        scale_t = scale_t.view(-1, 1, 1, 1)
        dm_xt = fm_xt * scale_t
        return dm_xt

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )
    
    def predict_start_from_eps(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def get_vector_field_from_v(self, v, x_t, t):
        """
        v is the SD v-parameterized vector field with v = sqrt(alpha_cumprod) * eps - sqrt(1 - alpha_cumprod) * z
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the v-parameterized vector field
        """
        z_pred = self.predict_start_from_z_and_v(x_t, t, v)
        eps_pred = self.predict_eps_from_z_and_v(x_t, t, v)
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field
    
    def get_vector_field_from_eps(self, noise, x_t, t):
        """
        eps is the SD eps-parameterized vector field with
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the eps-parameterized vector field
        """
        z_pred = self.predict_start_from_eps(x_t, t, noise)
        eps_pred = noise
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field
    
    def forward(self, x, t, **kwargs):
        """
        Forward pass for the flow model
        """
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t, **kwargs)
        return _pred


    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.sample_vt(fm_x=xt, fm_t=t, **cond_kwargs)

        return (vt - ut).square()