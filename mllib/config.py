from dataclasses import dataclass, field
import torch
import torch.nn as nn

@dataclass
class Mainmodel:
    task: str
    path: str
    class_name: str
    max_epsd: int
    outdir: str = './runs_train'
    enable_fp16: bool = True
    seed: int = 35
    
@dataclass
class ModelParams:
    no_param: bool
    in_shape: list[int] = field(default_factory=list)

@dataclass
class ModelDisc:
    name: str
    params: list[str] = field(default_factory=list)
    enable_train: bool = False
    opt_model: str = None
    model_params: ModelParams = None
    _params_curr: str = None
    _module: nn.Module = None

@dataclass
class Model:
    comp: dict[str, ModelDisc]
    root: str = None
    device: str = 'cuda'
    _device: torch.device = None
    _loss: dict[str, nn.Module] = field(default_factory=dict)
    _optimizer: torch.optim.Optimizer = None
    

@dataclass
class EnvParams:
    enable_render: bool
    difficulty: str = 'nomal'
    _observation_space: list[int] = field(default_factory=list)
    _action_space: list[int] = field(default_factory=list)

@dataclass
class Env:
    name: str
    enable_finish_epsd: bool = True
    enalbe_render: bool = True
    max_step: int = None
    env_params: EnvParams = None
    
@dataclass
class OptParams:
    lr: float
    wd: float = 0.0
    adam_eps: float = 1e-8
    adam_betas: list[float] = field(default_factory=lambda:[0.9, 0.999])
    
@dataclass
class Opt:
    name: str
    profile_name: str
    opt_params: OptParams
    _module: torch.optim.Optimizer = None
    
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
    opt: dict[str, Opt]
    log: Log
    info: Info