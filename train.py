import os
import sys
import hydra
import wandb
import torch
import signal
import datetime
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# ddp stuff
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from diff2flow.helpers import count_params, exists
from diff2flow.helpers import instantiate_from_config
from diff2flow.trainer_module import TrainerModuleLatentFM
from diff2flow.helpers import load_model_weights, load_lora_weights

torch.set_float32_matmul_precision('high')


def check_config(cfg):
    if cfg.get("auto_requeue", False):
        raise NotImplementedError("Auto-requeuing not working yet!")
    if exists(cfg.get("resume_checkpoint", None)) and exists(cfg.get("load_weights", None)):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")
    if "experiment" in cfg:
        raise ValueError("Experiment config not merged successfully!")
    if cfg.use_wandb and cfg.use_wandb_offline:
        raise ValueError("Decide either for Online and offline wandb, not both.")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(2024)

    """ Check config """
    check_config(cfg)

    """ Setup Logging """
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    postfix = f"{cfg.slurm_id}_{now}" if exists(cfg.slurm_id) else now
    exp_name = f"{cfg.name}_{postfix}" if exists(cfg.name) else postfix
    log_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    
    # setup loggers
    use_wandb_logging = cfg.use_wandb or cfg.use_wandb_offline
    if use_wandb_logging:
        usr_name = os.environ.get('USER', os.environ.get('USERNAME'))
        mode = "offline" if cfg.use_wandb_offline else "online"
        online_logger = WandbLogger(
            dir=log_dir,
            save_dir=log_dir,
            name=exp_name,
            project="lora-fm",
            tags=[usr_name, *cfg.get("tags", [])],
            config=OmegaConf.to_object(cfg),
            mode=mode,
            group="DDP"
        )
    else:
        online_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
            version="",
            log_graph=False,
            default_hp_metric=False,
        )
    csv_logger = CSVLogger(
        log_dir,
        name="",
        version="",
        prefix="",
        flush_logs_every_n_steps=500
    )
    csv_logger.log_hyperparams(OmegaConf.to_container(cfg))
    logger = [online_logger, csv_logger]

    """ Setup dataloader """
    data = instantiate_from_config(cfg.data)

    """ Setup model """
    trainer_module_params = dict(   
        # model
        fm_cfg              = cfg.model.fm_cfg,
        noising_step        = cfg.model.noising_step,
        start_from_noise    = cfg.model.start_from_noise,
        # first stage
        first_stage         = cfg.autoencoder,
        # lora
        lora_cfg            = cfg.lora.lora_cfg if "lora" in cfg else None,
        # conditioning
        cond_stage_cfg      = cfg.task.cond_stage_cfg,
        context_key         = cfg.task.context_key,
        conditioning_key    = cfg.task.conditioning_key,
        cond_dropout        = cfg.task.cond_dropout,
        # training
        lr                  = cfg.train.lr,
        weight_decay        = cfg.train.weight_decay,
        sampling_steps      = cfg.train.get("sampling_steps", 50),
        ema_rate            = cfg.train.ema_rate,
        ema_update_every    = cfg.train.ema_update_every,
        ema_update_after_step = cfg.train.ema_update_after_step,
        use_ema_for_sampling= cfg.train.use_ema_for_sampling,
        lr_scheduler_cfg    = cfg.train.lr_scheduler,
        # logging
        n_images_to_vis     = cfg.train.n_images_to_vis,
        metric_tracker_cfg  = cfg.task.metric_tracker_cfg,
        visualizer          = cfg.task.visualizer,
    )
    module = TrainerModuleLatentFM(**trainer_module_params)

    """ Setup callbacks """
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:06d}",
        # from config
        **cfg.train.checkpoint_callback_params
    )
    callbacks = [checkpoint_callback]
    
    # add tqdm progress bar callback
    if cfg.tqdm_refresh_rate != 1:
        from pytorch_lightning.callbacks import TQDMProgressBar
        tqdm_callback = TQDMProgressBar(refresh_rate=cfg.tqdm_refresh_rate)
        callbacks.append(tqdm_callback)

    # other callbacks from config
    callbacks_cfg = cfg.train.get("callbacks", None)
    if exists(callbacks_cfg):
        for cb_cfg in callbacks_cfg:
            cb = instantiate_from_config(cb_cfg)
            callbacks.append(cb)
    
    """ Setup trainer """
    if torch.cuda.is_available():
        print("Using GPU")
        gpu_kwargs = {'accelerator': 'gpu', 'strategy': 'ddp'}
        if cfg.devices > 0:
            gpu_kwargs["devices"] = cfg.devices
        else:       # determine automatically
            gpu_kwargs["devices"] = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
        gpu_kwargs["num_nodes"] = cfg.num_nodes
        if cfg.num_nodes >= 2:
            if cfg.deepspeed_stage > 0:
                gpu_kwargs["strategy"] = f'deepspeed_stage_{cfg.deepspeed_stage}'
            else:
                # multi-node hacks from
                # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
                gpu_kwargs["strategy"] = DDPStrategy(
                    gradient_as_bucket_view=True,
                    ddp_comm_hook=default_hooks.fp16_compress_hook
                )
        if cfg.auto_requeue:
            gpu_kwargs["plugins"] = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]
        if cfg.p2p_disable:
            # multi-gpu hack for heidelberg servers
            os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        print("Using CPU")
        gpu_kwargs = {'accelerator': 'cpu'}

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **gpu_kwargs,
        # from config
        **OmegaConf.to_container(cfg.train.trainer_params)
    )
    
    """ Setup signal handler """

    # hacky way to avoid define this in the traininer module
    def stop_training_method():
        module.stop_training = False
        print("-" * 40)
        print("Try to save checkpoint to {}".format(ckpt_dir))
        module.trainer.save_checkpoint(os.path.join(ckpt_dir, "interrupted.ckpt"))
        module.trainer.should_stop = True
        module.trainer.limit_val_batches = 0
        print("Saved checkpoint.")
        print("-" * 40)

    module.stop_training_method = stop_training_method

    # once the signal was sent, the stop_training flag tells
    # the pl module get ready for save checkpoint
    def signal_handler(sig, frame):
        print(f"Activate signal handler for signal {sig}")
        module.stop_training = True

    signal.signal(signal.SIGUSR1, signal_handler)

    """ Log some information """
    # compute global batchsize
    bs = cfg.data.params.batch_size
    bs = bs * gpu_kwargs["devices"]
    bs = bs * gpu_kwargs["num_nodes"]
    bs = bs * cfg.train.trainer_params.get("accumulate_grad_batches", 1)
    # log info
    some_info = {
        'Name': exp_name,
        'Log dir': log_dir,
        'Logging': "Wandb" if use_wandb_logging else "Tensorboard",
        'Model': cfg.model.fm_cfg.get("target", "not-specified"),
        'LoRA': cfg.lora.lora_cfg.lora_type if 'lora' in cfg else 'None',
        'Params': count_params(module),
        'Task': cfg.task.get("name", "not set"),
        'Data': cfg.data.get("name", "not set"),
        'Batchsize': cfg.data.params.batch_size,
        'Devices': gpu_kwargs["devices"],
        'Num nodes': gpu_kwargs["num_nodes"],
        'Gradient accum': cfg.train.trainer_params.get("accumulate_grad_batches", 1),
        'Global batchsize': bs,
        'Learning rate': cfg.train.lr,
        'Resume ckpt': cfg.resume_checkpoint,
        'Load weights': cfg.load_weights,
        'First stage': cfg.autoencoder.get("name", "not set") if "autoencoder" in cfg else None,
    }
    
    # Make sure we don't log multiple times
    if trainer.global_rank == 0:
        print("-" * 40)
        for k, v in gpu_kwargs.items():
            print(f"{k:<16}: {v}")
        print("-" * 40)
        for k, v in some_info.items():
            if use_wandb_logging:
                online_logger.experiment.summary[k] = v
            if isinstance(v, float):
                print(f"{k:<16}: {v:.5f}")
            elif isinstance(v, int):
                print(f"{k:<16}: {v:,}")
            elif isinstance(v, bool):
                print(f"{k:<16}: {'True' if v else 'False'}")
            else:
                print(f"{k:<16}: {v}")
        print("-" * 40)
        # log called command
        if use_wandb_logging:
            online_logger.experiment.summary["command"] = " ".join(["python"] + sys.argv)
        
        # save config file
        OmegaConf.save(cfg, f"{log_dir}/config.yaml")

    """ Train """
    ckpt_path = cfg.resume_checkpoint if exists(cfg.resume_checkpoint) else None
    if exists(cfg.load_weights):
        module = load_model_weights(module, cfg.load_weights, strict=True)
    if exists(cfg.load_lora_weights):
        module = load_lora_weights(module, cfg.load_lora_weights, strict=False)
    trainer.fit(module, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
