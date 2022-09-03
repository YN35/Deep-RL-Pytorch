import argparse
import yaml
from datetime import datetime
import shutil
from pathlib import Path
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

    return config

def load_config_and_make_output_directory(default_config, output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config)
    config_path = parser.parse_args().config

    # Load the config yaml as a Config object.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = from_dict(data_class=Config, data=config)

    # Make an output directory.
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir = str(Path(output_dir).joinpath(f'{config.task}', f'{now}'))
    config.misc.output_dir = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=False)

    # Copy the config yaml to the output directory.
    shutil.copy2(config_path, output_dir)

    return config

def load_main_model(cfg):
    if 'dqn' == cfg.task:
        from mllib.main_models.dqn import DQN as MainModel
    return MainModel(cfg)

def load_models(cfg):
    get_module_class(network.name)(dim_emb_position=positional_encoding._module.dim_feat)
    
    
def get_module_class(name):
    return getattr(getattr(__import__('mllib'), 'model'), name)