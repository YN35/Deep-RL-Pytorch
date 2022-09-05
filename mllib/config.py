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
    enable_fp16: bool = True
    seed: int = 35
    

@dataclass
class ModelDisc:
    name: str
    params: list[str] = field(default_factory=list)
    enable_train: bool = False
    lr: float = 0.05
    _params_curr: str = None
    _module: nn.Module = None
    in_shape: list[int] = field(default_factory=list)

@dataclass
class Model:
    comp: dict[str, ModelDisc]
    root: str = None
    device: str = 'cuda'
    _device: torch.device = None
    _loss: dict[str, nn.Module] = field(default_factory=dict)
    _optimizer: torch.optim.Optimizer = None
    

@dataclass
class Env:
    name: str
    enable_finish_epsd: bool = True
    max_epsd: int = None
    
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
class Info:
    blank: str = 'blank'
    config_path: str = None

@dataclass
class Config:
    mainmodel: Mainmodel
    model: Model
    env: Env
    opt: Opt
    log: Log
    info: Info