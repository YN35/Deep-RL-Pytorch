import argparse
import yaml
import importlib
from dacite import from_dict

from mllib.config import Config


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
    get_module_class(network.name)(dim_emb_position=positional_encoding._module.dim_feat)
    
    
def get_module_class(name):
    return getattr(getattr(__import__('mllib'), 'model'), name)