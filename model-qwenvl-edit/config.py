import yaml
import os
from typing import Any

def load_config() -> Any:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    # storage paths that don't begin with / are assumed to be relative to the config file
    filedir = os.path.dirname(os.path.abspath(__file__))
    for key in config['storage']:
        if not config['storage'][key].startswith('/'):
            config['storage'][key] = os.path.join(filedir, config['storage'][key])
    return config

config = load_config()
