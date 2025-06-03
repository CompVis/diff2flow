import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from diff2flow.helpers import un_normalize_ims

from diff2flow.dataset.depth_utils import apply_scale_and_shift
from diff2flow.dataset.depth_utils import abs_rel_error, delta1_accuracy


def calculate_PSNR(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnrs = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnrs.mean()


class ImageMetricTracker(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ssim = SSIM(data_range=1.)
        self.ssims = []

        self.psnrs = []

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )
        
    def __call__(self, target, pred):
        """ Assumes target and pred in [-1, 1] range """
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)

        # update FID
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        # SSIM and PSNR
        self.ssims.append(self.ssim(pred/2+0.5, target/2+0.5))
        self.psnrs.append(calculate_PSNR(pred/2+0.5, target/2+0.5))

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.fid.reset()

    def aggregate(self):
        fid = self.fid.compute()
        ssim = torch.stack(self.ssims).mean()
        psnr = torch.stack(self.psnrs).mean()
        out = dict(fid=fid, ssim=ssim, psnr=psnr)
        return out


class DepthMetricTracker:
    def __init__(self):
        super().__init__()
        self.delta1s = []
        self.relabs = []

        self.quantiles = (0.795, 46.679)      # DIODE quantiles

    def __call__(self, target, pred):
        """ Assumes target and pred in [-1, 1] range """
        target = target.mean(dim=1, keepdim=True)
        pred = pred.mean(dim=1, keepdim=True)

        # unnormalize ground truth depth
        target = target / 2 + 0.5
        q_min, q_max = self.quantiles
        q_min = torch.log(torch.tensor(q_min))
        q_max = torch.log(torch.tensor(q_max))
        log_target = target * (q_max - q_min) + q_min
        target = torch.exp(log_target)

        # scale and shift invariance
        pred = apply_scale_and_shift(pred=pred, gt=log_target)
        pred = pred.exp()

        # compute metrics
        relabs = abs_rel_error(pred, target)
        self.relabs.append(relabs)
        delta1 = delta1_accuracy(pred, target)
        self.delta1s.append(delta1)

    def reset(self):
        self.delta1s = []
        self.relabs = []

    def aggregate(self):
        delta1 = torch.stack(self.delta1s).mean()
        relabs = torch.stack(self.relabs).mean()
        out = dict(delta1=delta1, relabs=relabs)
        return out
