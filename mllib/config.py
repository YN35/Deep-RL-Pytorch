from dataclasses import dataclass, field
from torch.utils.data import Dataset
import torch
import torch.nn as nn

@dataclass
class Callable:  # Typically a function or a class.
    name: str
    args: dict = field(default_factory=dict)

@dataclass
class ModelDisc:
    name: str
    params: list[str] = field(default_factory=list)
    no_snapshot: bool = False
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
class EnvDisc:
    name: str
    path: str
    dataset_class: str
    transform: str
    _path: str = None

@dataclass
class Division:
    envs: list[EnvDisc] = field(default_factory=list)
    minibatch_size: int = 256  # Note that lr = base_initial_lr * minibatch_size / 256.
    num_workers: int = 1
    drop_last: bool = True  # Only applied to the train dataset.
    shuffle: bool = True  # Only applied to the train dataset.

@dataclass
class Dataset:
    root: str
    divisions: dict[str, Division] = field(default_factory=dict)

@dataclass
class LearningSchedule:
    base_initial_lr: float = 1.0e-5
    gain_threshold_turn_phase: float = 0.0
    gain_threshold_warmup_phase: float = 1.0e-5
    lr_warmup_ratio: float = 1.25
    gain_window_size_warmup_phase: int = 5
    gain_threshold_cooldown_phase: float = 1.0e-6
    lr_cooldown_ratio: float = 0.95
    gain_window_size_cooldown_phase: int = 100
    suspend_cooldown_threshold_ratio: float = 0.9
    dpts_train: int = 1000000000
    dpts_eval: int = 100000
    dpts_snapshot: int = 100000
    dpts_train_log: int = 10000
    dpts_update_lr: int = 10000
    skip_init_eval: bool = False
    _initial_lr: float = 0
    _lr: float = 0
    _is_warmup_phase: bool = True
    _suspend_cooldown: bool = True
    _gain_stat: list = None

# TODO: messy...
@dataclass
class Misc:
    output_dir: str = './output'
    seed: int = 42
    grad_clip_value: float = -1.0
    grad_clip_norm: float = -1.0
    wd: float = 0.0
    adam_betas: list[float] = field(default_factory=lambda:[0.9, 0.999])
    adam_eps: float = 1e-8
    lambda_lipschitz: float = 0.0
    lip_c: float = 0.0
    classes: list[list[str]] = field(default_factory=list)
    smoothing: float = 0.0
    flooding: float = 0.0
    cam_sparsity: float = 0.0
    cam_boundary_mask_pixels: int = 0
    cam_mask_value: float = -20
    heatmap_gaussian_std: int = 8
    heatmap_mp_kernel_size: int = 5
    heatmap_peak_threshold: float = 0.75
    keypoint_match_error_threshold: float = 1.0

@dataclass
class InverseRendering:
    env_emit: list[float] = field(default_factory=lambda:[1.0, 1.0, 1.0])
    randomize_env_emit: bool = False
    train_image_batch_size: int = 0
    train_ray_batch_size: int = 8192
    test_ray_batch_size: int = 8192
    near: float = 0.01
    far: float = 1.0
    grid_div: int = 128
    num_importance_samples: int = 32
    importance_sampling_max_div: int = 4
    # TODO: move to the model arguments?
    hash_embedding_num_resolutions: list[int] = field(default_factory=lambda:[8,])
    hash_embedding_dim_feat_per_resolution: list[int] = field(default_factory=lambda:[4,])
    hash_embedding_coarsest_grid_div: list[int] = field(default_factory=lambda:[32,])
    hash_embedding_finest_grid_div: list[int] = field(default_factory=lambda:[2048,])
    dim_pos_feat: int = 64
    init_opacity: float = 0.5
    init_emit: float = 0.5
    init_inv_beta: float = 1.0
    gamma_correction: float = 1.0
    diffuse: bool = False
    reflect: bool = False
    concat_normal: bool = False
    neus: bool = False
    surface_ext_threshold: float = 0.5

    random_offset: float = 1.0
    random_perturb: float = 0.0
    #ray_skip_ext_threshold: float = 0.0
    #ray_skip_transmit_threshold: float = 0.0

    lambda_volume_rendering: float = 1.0
    lambda_ray_entropy: float = 0.0
    lambda_normal: float = 0.0
    lambda_ext: float = 0.0
    lambda_emit: float = 0.0
    lambda_eikonal: float = 1.0e-9

    position_texture_path: str = ''

    ###PhySG
    sg_num: int = 128
    gamma: float = 1.0
    train_cameras: bool =  False

@dataclass
class Config:
    task: str
    model: Model
    dataset: Dataset
    ls: LearningSchedule = field(default_factory=LearningSchedule)
    misc: Misc = field(default_factory=Misc)
    inverse_rendering: InverseRendering = field(default_factory=InverseRendering)