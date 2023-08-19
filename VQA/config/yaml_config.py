from pprint import pformat
import yaml

class Config:
    """
    Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def get_config(yaml_filepath: str) -> Config:
    with open(yaml_filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

