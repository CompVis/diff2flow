import wandb
import math
import einops
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from diff2flow.ema import EMA

from diff2flow.helpers import exists
from diff2flow.helpers import freeze
from diff2flow.helpers import resize_ims
from diff2flow.helpers import instantiate_from_config
from diff2flow.helpers import load_partial_from_config

from diff2flow.lora import LoraLinear, LoRAConv # LoRA
from diff2flow.lora import LoRAAdapterConv, DataProvider # LoRAAdapter
from diff2flow.lora import getattr_recursive, setattr_recursive

from diff2flow.diffusion import ForwardDiffusion


class TrainerModuleLatentFM(LightningModule):
    def __init__(
        self,
        # models
        fm_cfg: dict,
        noising_step: int = -1,
        start_from_noise: bool = False,
        # first stage
        first_stage: dict = None,
        # lora
        lora_cfg: dict = None,
        # conditioning
        cond_stage_cfg: dict = None,
        context_key: str = None,
        cond_dropout: float = 0.0,
        conditioning_key: str = None,
        # training
        lr: float = 1e-4,
        weight_decay: float = 0.,
        sampling_steps: int = 50,
        ema_rate: float = 0.999,
        ema_update_every: int = 100,
        ema_update_after_step: int = 1000,
        use_ema_for_sampling: bool = True,
        lr_scheduler_cfg: dict = None,
        # logging
        n_images_to_vis: int = 16,
        log_grad_norm: bool = False,
        metric_tracker_cfg: dict = None,
        visualizer: dict = None,
    ):
        """
        Args:
            fm_cfg: Flow matching model config.
            noising_step: Forward diffusion noising step with linear schedule
                of Ho et al. Set to -1 to disable.
            start_from_noise: Whether to start from noise with low-res image as
                conditioning (FM) or directly from low-res image (IC-FM).
            first_stage: First stage config, if None, identity is used.
            lora_cfg: LoRA config, if None, no LoRA is used.
            cond_stage_cfg: Conditioning stage config, if None, no conditioning is used.
            context_key: Context conditioning signal, concatenated to the input.
            conditioning_key: Key in the batch to use for conditioning.
            cond_dropout: Dropout rate for conditioning.
            lr: Learning rate.
            weight_decay: Weight decay.
            sampling_steps: Number of sampling steps for inference.
            ema_rate: EMA rate.
            ema_update_every: EMA update rate (every n steps).
            ema_update_after_step: EMA update start after n steps.
            use_ema_for_sampling: Whether to use the EMA model for sampling.
            lr_scheduler_cfg: Learning rate scheduler config.
            n_images_to_vis: Number of images to visualize.
            log_grad_norm: Whether to log gradient norm.
            metric_tracker_cfg: Metric tracker config, if None, no metrics are tracked.
            visualizer: Visualizer config, if None, no visualization is done.
        """
        super().__init__()
        self.model = instantiate_from_config(fm_cfg)

        # lora
        self.lora_cfg = lora_cfg
        if self.lora_cfg is not None:
            self.lora_type = self.lora_cfg["lora_type"]
            self.add_lora_to_unet()
        else:
            self.lora_type = None

        if ema_rate == 0.0:
            self.ema_model = None
            assert not use_ema_for_sampling, "Cannot use EMA for sampling without EMA model"
        else:
            self.ema_model = EMA(
                self.model, beta=ema_rate,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
                power=3/4.,                     # recommended for trainings < 1M steps
                include_online_model=False      # we have the online model stored here
            )
        self.use_ema_for_sampling = use_ema_for_sampling

        # forward diffusion of image
        self.noise_image = noising_step > 0
        self.noising_step = noising_step
        self.diffusion = ForwardDiffusion() if self.noising_step > 0 else None
            
        self.cond_dropout = cond_dropout
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        # first stage encoding
        if first_stage is not None:
            self.scale_factor = first_stage.get("scale_factor", 1.0)
            self.first_stage = instantiate_from_config(first_stage.first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval()
            if self.scale_factor == 1.0:
                import warnings
                warnings.warn("Using first stage with scale_factor=1.0")
        else:
            if self.scale_factor != 1.0:
                raise ValueError("Cannot use scale_factor with identity first stage")
            self.first_stage = None

        # conditioning
        self.start_from_noise = start_from_noise
        if self.start_from_noise and self.noise_image:
            raise ValueError("Cannot use noising step with start_from_noise=True")
        self.context_key = context_key
        self.conditioning_key = conditioning_key

        # cross attention encoder
        if cond_stage_cfg is not None:
            assert conditioning_key is not None, "Need conditioning key for cond_stage!"
            self.cond_stage = instantiate_from_config(cond_stage_cfg)
            freeze(self.cond_stage)
            self.cond_stage.eval()
            if hasattr(self.cond_stage, "get_unconditional_conditioning"):
                self.uncond = self.cond_stage.get_unconditional_conditioning(device=self.device)
                if len(self.uncond.shape) > 1:
                    self.uncond = self.uncond.squeeze(0)
            else:
                self.uncond = None
        else:
            self.cond_stage = None
        assert not (self.cond_stage is None and self.cond_dropout > 0.), "Cannot use cond_dropout without cond_stage"
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.sampling_steps = sampling_steps

        self.validation_samples = None
        self.n_images_to_vis = n_images_to_vis
        self.val_epochs = 0

        if metric_tracker_cfg is not None:
            self.metric_tracker = instantiate_from_config(metric_tracker_cfg)
            if isinstance(self.metric_tracker, nn.Module):
                self.metric_tracker = self.metric_tracker.to(self.device)
        else:
            self.metric_tracker = None

        if visualizer is not None:
            self.visualizer = instantiate_from_config(visualizer)
        else:
            self.visualizer = None

        self.save_hyperparameters()

        # flag to make sure the signal is not handled at an
        # incorrect state, e.g. during weights update
        self.stop_training = False

    # dummy function to be compatible
    def stop_training_method(self):
        pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out

    def forward(self, x1: Tensor, x0: Tensor, **kwargs):
        return self.model.training_losses(x1=x1, x0=x0, **kwargs).mean()

    def add_lora_to_unet(self):
        unet = self.model
        freeze(unet)
        self.params_to_optimize = []
        self.params_names = []
        lora_cfg = self.lora_cfg

        assert lora_cfg is not None, "LoRA config cannot be None"
        do_lora_conv = "lora_conv" in lora_cfg
        do_first_full_conv = lora_cfg.get("do_full_first_conv", False)
        do_lora_self_attn = "lora_self_attn" in lora_cfg
        do_lora_cross_attn = "lora_cross_attn" in lora_cfg
        do_lora_full_attn = "lora_full_attn" in lora_cfg
        do_lora_attn = do_lora_self_attn or do_lora_cross_attn or do_lora_full_attn
        do_lora_mlp = "lora_mlp" in lora_cfg
        assert not (do_lora_full_attn and (do_lora_self_attn or do_lora_cross_attn)), "LoRA full attn cannot be used with self or cross attn"
        assert do_lora_conv or do_lora_attn, "LoRA config must contain either 'lora_conv' or 'lora_attn'"

        lora_scale = lora_cfg.get("lora_scale", 1.0)
        if self.lora_type == "lora_adapter":
            self.conv_data_provider_names = []
            lora_cdim = lora_cfg.get("lora_cdim", 4)
        
        # LEGACY: _rank and _ratio keys shouldnt be in the general config file!
        lora_cfg_cp = {k: v for k, v in lora_cfg.items() if not k.endswith("_ratio") and not k.endswith("_rank")}
        # Populate keys
        for k in list(lora_cfg_cp.keys()):
            # if k not in ["lora_conv", "lora_self_attn", "lora_cross_attn", "lora_full_attn", "do_full_first_conv", "lora_scale", "do_full_last_conv"]:
            #     raise ValueError(f"Unknown LoRA config key {k}")
            # if not isinstance(lora_cfg_cp[k], (float, int)):
            #     raise ValueError(f"LoRA config key must be float or int")
            if isinstance(lora_cfg_cp[k], float):
                lora_cfg_cp[k+"_ratio"] = lora_cfg_cp[k]
            elif isinstance(lora_cfg_cp[k], int):
                lora_cfg_cp[k+"_rank"] = lora_cfg_cp[k]
        
        # train last layer fully
        do_full_last_conv = lora_cfg_cp.get("do_full_last_conv", False)
        if do_full_last_conv:
            self.params_to_optimize.extend([p for p in self.model.net.out.parameters()])
            self.model.net.out[0].weight.requires_grad = True
            self.model.net.out[0].bias.requires_grad = True
            self.model.net.out[2].weight.requires_grad = True
            self.model.net.out[2].bias.requires_grad = True

        for path, w in unet.state_dict().items():
            if "." not in path:
                continue
            # Determine whether we want to finetune the first full convolutional layer
            if "net.input_blocks.0.0" in path and do_first_full_conv:
                self.params_to_optimize.append(w)
                self.model.net.input_blocks[0][0].weight.requires_grad = True
                self.model.net.input_blocks[0][0].bias.requires_grad = True
            
            # Add LoRA layers to the full attention modules
            elif "attn" in path and do_lora_full_attn:
                if path.split(".")[-2] not in ["to_q", "to_k", "to_v", "query", "key", "value"]:
                    continue
                print(f"Adding attention LoRA to {path}")
                
                attn_module = getattr_recursive(unet, ".".join(path.split(".")[:-2]))
                layer_module = getattr_recursive(unet, ".".join(path.split(".")[:-1]))
                ll = LoraLinear(
                    layer_module.out_features,
                    layer_module.in_features,
                    lora_cfg_cp.get("lora_full_attn_rank",
                                 math.ceil((max(layer_module.out_features, 
                                                layer_module.in_features) * 
                                                lora_cfg_cp.get("lora_full_attn_ratio", -1))),
                                ),
                    lora_scale,
                )
                # W is the original weight matrix
                ll.W.load_state_dict({path.split(".")[-1]: w})
                setattr(
                    attn_module,
                    path.split(".")[-2],
                    ll,
                )
                for p in ll.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
                        self.params_names.append(path)

            # Add LoRA layers to the self attention modules
            elif "attn1" in path and do_lora_self_attn:
                if path.split(".")[-2] not in ["to_q", "to_k", "to_v", "query", "key", "value"]:
                    continue
                print(f"Adding attention LoRA to {path}")
                
                attn_module = getattr_recursive(unet, ".".join(path.split(".")[:-2]))
                layer_module = getattr_recursive(unet, ".".join(path.split(".")[:-1]))
                ll = LoraLinear(
                    layer_module.out_features,
                    layer_module.in_features,
                    lora_cfg_cp.get("lora_self_attn_rank",
                                 math.ceil((max(layer_module.out_features, 
                                                layer_module.in_features) * 
                                                lora_cfg_cp.get("lora_self_attn_ratio", -1))),
                                ),
                    lora_scale,
                )
                # W is the original weight matrix
                ll.W.load_state_dict({path.split(".")[-1]: w})
                setattr(
                    attn_module,
                    path.split(".")[-2],
                    ll,
                )
                for p in ll.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
                        self.params_names.append(path)

            # Add LoRA layers to the cross attention modules
            elif "attn2" in path and do_lora_cross_attn:
                if path.split(".")[-2] not in ["to_q", "to_k", "to_v", "query", "key", "value"]:
                    continue
                print(f"Adding attention LoRA to {path}")
                
                attn_module = getattr_recursive(unet, ".".join(path.split(".")[:-2]))
                layer_module = getattr_recursive(unet, ".".join(path.split(".")[:-1]))
                ll = LoraLinear(
                    layer_module.out_features,
                    layer_module.in_features,
                    lora_cfg_cp.get("lora_cross_attn_rank",
                                 math.ceil((max(layer_module.out_features, 
                                                layer_module.in_features) * 
                                                lora_cfg_cp.get("lora_cross_attn_ratio", -1))),
                                ),
                    lora_scale,
                )
                # W is the original weight matrix
                ll.W.load_state_dict({path.split(".")[-1]: w})
                setattr(
                    attn_module,
                    path.split(".")[-2],
                    ll,
                )
                for p in ll.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
                        self.params_names.append(path)
            
            # Add LoRA layers to the convolutional layers
            elif "bias" in path:
                continue

            elif isinstance(getattr_recursive(unet, ".".join(path.split(".")[:-1])), torch.nn.Conv2d) and do_lora_conv:
                # add exception for full finetuning of last layer
                if do_full_last_conv and (("net.out.0" in path) or ("net.out.2" in path)):
                    print(f"Skipping {path} for full last conv finetuning")
                    continue
                print(f"Adding convolutional LoRA to {path}")
                conv_module = getattr_recursive(unet, ".".join(path.split(".")[:-1]))
                if self.lora_type == "lora":
                    ll = LoRAConv(
                        conv_module.in_channels,
                        conv_module.out_channels,
                        conv_module.kernel_size,
                        conv_module.stride,
                        conv_module.padding,
                        lora_cfg_cp.get("lora_conv_rank",
                                        math.ceil(max(conv_module.in_channels, 
                                                    conv_module.out_channels) * 
                                                    lora_cfg_cp.get("lora_conv_ratio", -1)),
                                    ),
                        lora_scale,
                    )
                elif self.lora_type == "lora_adapter":
                    ll = LoRAAdapterConv(
                        conv_module.in_channels,
                        conv_module.out_channels,
                        conv_module.kernel_size,
                        conv_module.stride,
                        conv_module.padding,
                        DataProvider(),
                        lora_cdim,
                        lora_cfg_cp.get("lora_conv_rank",
                                        math.ceil(max(conv_module.in_channels, 
                                                    conv_module.out_channels) * 
                                                    lora_cfg_cp.get("lora_conv_ratio", -1)),
                                    ),
                        lora_scale,
                    )
                    self.conv_data_provider_names.append(".".join(path.split(".")[:-1]) + ".data_provider")
                else:
                    raise ValueError(f"Unknown LoRA type {self.lora_type}")

                # Find the bias term
                bias_path = ".".join(path.split(".")[:-1] + ["bias"])
                b = getattr_recursive(unet, bias_path)
                ll.W.load_state_dict({path.split(".")[-1]: w, "bias": b})

                # swap conv_module with ll
                setattr_recursive(unet, ".".join(path.split(".")[:-1]), ll)

                for p in ll.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
                        self.params_names.append(path)

            # Add LoRA layers to the fully connected layers
            elif any([k in path for k in ["to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]]) and do_lora_mlp:
                print(f"Adding MLP LoRA to {path}")
                fc_module = getattr_recursive(unet, ".".join(path.split(".")[:-1]))
                ll = LoraLinear(
                    fc_module.out_features,
                    fc_module.in_features,
                    lora_cfg_cp.get("lora_mlp_rank",
                                 math.ceil((max(fc_module.out_features, 
                                                fc_module.in_features) * 
                                                lora_cfg_cp.get("lora_mlp_ratio", -1))),
                                ),
                    lora_scale,
                )
                # W is the original weight matrix
                ll.W.load_state_dict({path.split(".")[-1]: w})
                setattr_recursive(unet, ".".join(path.split(".")[:-1]), ll)
                for p in ll.parameters():
                    if p.requires_grad:
                        self.params_to_optimize.append(p)
                        self.params_names.append(path)
        
        num_params = sum(p.numel() for p in self.params_to_optimize)
        print(f"[LoRA] Optimizing {len(self.params_to_optimize)} LoRA layers and {num_params/1e6:.2f}M LoRA parameters")

    @torch.no_grad()
    def encode_first_stage(self, x):
        if exists(self.first_stage):
            x = self.first_stage.encode(x)
            if not isinstance(x, torch.Tensor): # hack for posterior of original VAE
                x = x.mode()
            x = x * self.scale_factor
        return x

    @torch.no_grad()
    def decode_first_stage(self, z: Tensor):
        if exists(self.first_stage):
            z = z / self.scale_factor
            z = self.first_stage.decode(z)
        return z
   
    def apply_cond_dropout(self, x):
        if self.training and self.cond_dropout > 0.:
            mask = torch.bernoulli(torch.ones(x.shape[0]) * self.cond_dropout).bool()
            if self.uncond.device != x.device:
                self.uncond = self.uncond.to(x.device)
            x[mask] = self.uncond
        return x
    
    def extract_from_batch(self, batch):
        """
        Takes batch and extracts data.

        Returns:
            x0: Samples from source distribution (can also be None if we start from noise).
            x0_latent: Source latent codes. If identity first stage, this is the same as x0.
            x1: Samples from target distribution (must always be provided).
            x1_latent: Target latent codes. if identity first stage, this is the same as x1.
            mask: If available, a valid mask for the target.
            mask_latent: If available, latent valid mask for target.
        """
        # source samples (x0 ~ p(x0))
        if "x0" in batch:
            x0 = batch["x0"]
            # check for precomputed latents
            if "x0_latent" in batch:
                x0_latent = batch["x0_latent"]
                x0_latent = x0_latent * self.scale_factor
            else:
                x0_latent = self.encode_first_stage(x0)
        else:
            x0, x0_latent = None, None

        # target samples (x1 ~ p(x1)) - data
        x1 = batch["x1"]
        # check for precomputed latents
        if "x1_latent" in batch:
            x1_latent = batch["x1_latent"]
            x1_latent = x1_latent * self.scale_factor
        else:
            x1_latent = self.encode_first_stage(x1)

        # check for valid mask
        if "mask" in batch or "valid_mask" in batch:
            mask = batch["mask"] if "mask" in batch else batch["valid_mask"]
            # resize valid mask to latent space size
            mask_latent = resize_ims(mask.float(), size=x1_latent.shape[-2:], mode="bilinear")
            mask_latent = mask_latent == 1
        else:
            mask, mask_latent = None, None

        return {
            "x0": x0, "x0_latent": x0_latent,
            "x1": x1, "x1_latent": x1_latent,
            "mask": mask, "mask_latent": mask_latent
        }
    
    def training_step(self, batch, batch_idx):
        """ extract data """
        data = self.extract_from_batch(batch)
        x1_latent = data["x1_latent"]
        x0_latent = data["x0_latent"]

        """ cross-attention conditioning """
        if exists(self.cond_stage):
            # fetch conditioning from raw batch
            conditioning = batch[self.conditioning_key]
            conditioning = self.cond_stage(conditioning)
            conditioning = self.apply_cond_dropout(conditioning)
        else:
            conditioning = None
        
        """ concatenated context """
        if exists(self.context_key):
            # fetch context from preprocessed batch
            context = data[self.context_key]
        else:
            context = None

        """ input """
        x_target = x1_latent

        # define x0
        if self.start_from_noise:
            x_source = batch.get("noise", torch.randn_like(x1_latent))
        else:
            x_source = x0_latent

        # noise x0
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        """ loss """
        loss = self.forward(
            x0=x_source,
            x1=x_target,
            context=context,
            context_ca=conditioning
        )
        bs = x_source.shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=bs)

        """ log statistics """
        if exists(self.ema_model): self.ema_model.update()
        if self.stop_training: self.stop_training_method()
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # save samples for visualization
        if self.validation_samples is None:
            self.validation_samples = {
                k: (
                    v[:self.n_images_to_vis].clone()
                    if isinstance(v, Tensor) else v[:self.n_images_to_vis]
                )
                for k, v in batch.items()
            }

        """ extract data """
        data = self.extract_from_batch(batch)
        x1, x1_latent = data["x1"], data["x1_latent"]
        x0, x0_latent = data["x0"], data["x0_latent"]
    
        """ cross-attention conditioning """
        if exists(self.cond_stage):
            # fetch conditioning from raw batch
            conditioning = batch[self.conditioning_key]
            conditioning = self.cond_stage(conditioning)
        else:
            conditioning = None
        
        """ concatenated context """
        if exists(self.context_key):
            # fetch context from preprocessed batch
            context = data[self.context_key]
        else:
            context = None

        """ input """
        # define x0
        if self.start_from_noise:
            x_source = batch.get("noise", torch.randn_like(x1_latent))
        else:
            x_source = x0_latent

        # noise x0
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        """ prediction """
        model = self.ema_model if self.use_ema_for_sampling else self.model
        sample_kwargs = dict(num_steps=self.sampling_steps) if hasattr(self, "sampling_steps") else {}
        x1_latent_pred = model.generate(x_source, context=context, context_ca=conditioning, sample_kwargs=sample_kwargs)
        
        # decode
        x1_pred = self.decode_first_stage(x1_latent_pred)

        """ metrics """
        self.metric_tracker(x1, x1_pred)

        if self.stop_training:
            self.stop_training_method()
    
    # TODO: insert self.inference into validation_step and evaluate_and_visualize_batch
    # to avoid any inconsistencies.
    def inference(self, batch, use_ema: bool = True, **kwargs):
        # check for precomputed latents
        if "x0_latent" in batch:
            x0_latent = batch["x0_latent"]
            x0_latent = x0_latent * self.scale_factor
        elif "x0" in batch:
            x0 = batch["x0"]
            x0_latent = self.encode_first_stage(x0)
        else:
            # noise or x0/x0_latent required to obtain shape for inference
            assert (self.context_key != "x0_latent") & (self.context_key != "x0"), "x0_latent or x0 required for conditioning"
            assert self.start_from_noise, "Only models starting from noise are allowed without x0 and x0_latent"
            assert "noise" in batch, "Noise required for inference w/o x0 and x0_latent"
            noise = batch["noise"]
            x0_latent = noise       # ignored
        batch["x0_latent"] = x0_latent    # scaled (ignored if starting from noise)

        """ cross-attention conditioning """
        if exists(self.cond_stage):
            # fetch conditioning from raw batch
            conditioning = batch[self.conditioning_key]
            conditioning = self.cond_stage(conditioning)
        else:
            conditioning = None
        
        """ concatenated context """
        if exists(self.context_key):
            # fetch context from preprocessed batch
            context = batch[self.context_key]
        else:
            context = None
        
        """ input """
        # define x0
        if self.start_from_noise:
            x_source = batch.get("noise", torch.randn_like(x0_latent))
        else:
            x_source = x0_latent

        # noise x0
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        """ prediction """
        if use_ema:
            assert exists(self.ema_model), "Cannot use EMA for inference without EMA model"
        model = self.ema_model if use_ema else self.model
        x1_latent_pred = model.generate(
            x_source, context=context,
            context_ca=conditioning,
            **kwargs
        )
        
        # decode
        x1_pred = self.decode_first_stage(x1_latent_pred)

        return x1_pred

    def on_validation_epoch_end(self):
        self.evaluate_and_visualize_batch(self.validation_samples, prefix="val")

        # aggregate metrics
        if self.metric_tracker is not None:
            metrics = self.metric_tracker.aggregate()
            self.metric_tracker.reset()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)
        
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs} | Optimizer step {self.global_step}")
        torch.cuda.empty_cache()

    def evaluate_and_visualize_batch(self, batch, prefix="train"):
        data = self.extract_from_batch(batch)
        x1, x1_latent = data["x1"], data["x1_latent"]
        x0, x0_latent = data["x0"], data["x0_latent"]
        mask, mask_latent = data["mask"], data["mask_latent"]

        # cross-attention conditioning
        if exists(self.cond_stage):
            conditioning = batch[self.conditioning_key]
            conditioning = self.cond_stage(conditioning)
        else:
            conditioning = None

        # concatenated context
        if exists(self.context_key):
            context = data[self.context_key]
        else:
            context = None

        # input
        if self.start_from_noise:
            x_source = batch.get("noise", torch.randn_like(x1_latent))
        else:
            x_source = x0_latent

        # noise x0
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        # prediction
        model = self.ema_model if self.use_ema_for_sampling else self.model
        sample_kwargs = dict(num_steps=self.sampling_steps) if hasattr(self, "sampling_steps") else {}
        x1_latent_pred = model.generate(x_source, context=context, context_ca=conditioning, sample_kwargs=sample_kwargs)
        x1_pred = self.decode_first_stage(x1_latent_pred)

        # visualize
        if self.visualizer is not None:
            img = self.visualizer(x0, x1, x1_pred)
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"{prefix}/images": wandb.Image(img)}, step=self.global_step)
            else:
                img = einops.rearrange(np.array(img), 'h w c -> c h w')
                self.logger.experiment.add_image(f"{prefix}/images", img, global_step=self.global_step)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
