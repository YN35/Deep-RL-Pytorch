import argparse
import yaml
import importlib
from dacite import from_dict

import torch

from mllib.config import Config
from mllib.model import get_model_class
from mllib.env import get_env_class

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    config_path = parser.parse_args().config

    # Load the config yaml as a Config object.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = from_dict(data_class=Config, data=config)
    
    config.info.config_path = config_path

    return config

def load_main_model(cfg):
    """モデルのメインファイルを読み込む
    """
    # if 'dqn' == cfg.mainmodel:
    #     from mllib.main_models.dqn.main import DQN as MainModel
    module_path = cfg.mainmodel.path
    module = importlib.import_module(module_path)
    MainModel = getattr(module, cfg.mainmodel.class_name)
    return MainModel(cfg)

def load_models(cfg):
    """cfgから読み込んでモデルのインスタンスを_moduleに格納する

    Args:
        cfg ([dataclass]): [description]

    Returns:
        [dataclass]: [description]
    """
    for comp in cfg.model.comp.values():
        comp._module = get_model_class(comp.name)(comp.model_params).to(cfg.model._device)
    return cfg

def load_envs(cfg):
    """cfgから読み込んで環境のインスタンスを_moduleに格納する

    Args:
        cfg ([dataclass]): [description]

    Returns:
        [dataclass]: [description]
    """
    cfg.env._module = get_env_class(cfg.env.name)(cfg.env.env_params)
    return cfg

def set_device(cfg):
    cfg.model._device = torch.device(cfg.model.device) if cfg.model.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[Device] ' + str(cfg.model._device))
    return cfg
    

