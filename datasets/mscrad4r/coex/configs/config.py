from ruamel.yaml import YAML


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    if 'model' in cfg:
        if 'backbone' in cfg['model']['stereo']:
            backbone_cfg = YAML().load(
                open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
            cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg
