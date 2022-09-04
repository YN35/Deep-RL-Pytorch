from dataclasses import dataclass, field
from torch.utils.data import Dataset
import torch
import torch.nn as nn

@dataclass
class Mainmodel:
    task: str
    path: str
    class_name: str
    outdir: str = './runs_train'
    
@dataclass
class HyperParams:
    in_shape: tuple

@dataclass
class ModelDisc:
    name: str
    params: list[str] = field(default_factory=list)
    enable_train: bool
    lr: float
    hyper_params: HyperParams
    _params_curr: str = None
    _module: nn.Module = None

@dataclass
class Model:
    components: dict[str, ModelDisc]
    root: str = None
    device: str = 'cuda'
    _device: torch.device = None
    _loss: dict[str, nn.Module] = field(default_factory=dict)  # TODO: treat equally as components
    _optimizer: torch.optim.Optimizer = None
    

@dataclass
class Env:
    name: str
    
@dataclass
class Opt:
    name: str
    #weight decay
    wd: float = 0.0
    
@dataclass
class LogPrint:
    step: int = 100
    epsd: int = 10000000000
    
@dataclass
class LogBord:
    step: int = 100
    epsd: int = 1

@dataclass
class Log:
    print: LogPrint
    bord: LogBord

@dataclass
class Misc:
    seed: int = 35

@dataclass
class Config:
    mainmodel: Mainmodel
    model: Model
    env: Env
    opt: Opt
    log: Log