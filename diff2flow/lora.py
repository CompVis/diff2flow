import torch
from torch import nn
from typing import Union, Tuple, Any

def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split('.')
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

target_modules: list[str] = ["to_q", "to_k", "to_v", "query", "key", "value"]

def setattr_recursive(obj: Any, path: str, value: Any) -> None:
    parts = path.split('.')
    for part in parts[:-1]:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

class DataProvider:
    def __init__(self):
        self.batch = None

    def set_batch(self, batch):
        if self.batch is not None:
            if isinstance(self.batch, torch.Tensor):
                assert self.batch.shape[1:] == batch.shape[1:], "Check: shapes probably should not change during training"

        self.batch = batch

    def get_batch(self, x=None):
        assert self.batch is not None, "Error: need to set a batch first"

        if x is None or isinstance(self.batch, torch.Tensor):
            return self.batch

        # batch is a list; select the corresponding element based on x
        size = x.shape[2]
        for i in range(len(self.batch)):
            if self.batch[i].shape[2] == size:
                return self.batch[i]
            
        raise ValueError("Error: no matching batch found")

    def reset(self):
        self.batch = None

class LoraLinear(torch.nn.Module):
    def __init__(
        self,
        out_features,
        in_features,
        rank = None,
        lora_scale = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale

        # original weight of the matrix
        self.W = nn.Linear(in_features, out_features, bias=False)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # b should be init wiht 0
        for p in self.B.parameters():
            p.detach().zero_()

    def forward(self, x):
        w_out = self.W(x)
        a_out = self.A(x)
        b_out = self.B(a_out)
        return w_out + b_out * self.lora_scale
    

class LoRAConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        rank: int = None,
        lora_scale: float = 1.0,
    ):
        super().__init__()

        # self.lora_scale = alpha / rank
        self.lora_scale = lora_scale

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): In shape of (B, C, H, W)
        """
        w_out = self.W(x)
        a_out = self.A(x)
        b_out = self.B(a_out)

        return w_out + b_out * self.lora_scale
    

class LoRAAdapterConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        data_provider: DataProvider,
        c_dim: int,
        rank: int = None,
        lora_scale: float = 1.0,
    ):
        super().__init__()

        # self.lora_scale = alpha / rank
        self.lora_scale = lora_scale
        self.c_dim = c_dim

        self.data_provider = data_provider

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.beta = nn.Conv2d(c_dim, rank, kernel_size=1, bias=False)
        self.gamma = nn.Conv2d(c_dim, rank, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): In shape of (B, C, H, W)
        """
        w_out = self.W(x)
        a_out = self.A(x)

        # inject conditioning into LoRA
        c = self.data_provider.get_batch(a_out)
        element_shift = self.beta(c)
        element_scale = self.gamma(c) + 1
        a_cond = a_out * element_scale + element_shift

        b_out = self.B(a_cond)

        return w_out + b_out * self.lora_scale