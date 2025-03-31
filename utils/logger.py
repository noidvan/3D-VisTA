from torch.utils.tensorboard import SummaryWriter
from pipeline.registry import registry
import os
import wandb
import collections.abc

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

@registry.register_utils("tensorboard_logger")
class TensorboardLogger(object):
    def __init__(self, cfg):
        log_dir = cfg['logger']['args']['log_dir']
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir)
    
    def log(self, log_dict, step=None):
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, step)

@registry.register_utils("wandb_logger")
class WandbLogger(object):
    def __init__(self, cfg):
        # Get the log directory and project name from the config (with defaults)
        entity  = cfg['logger']['args'].get('entity', 'default_entity')
        log_dir = cfg['logger']['args'].get('log_dir', './wandb_logs')
        project = cfg['logger']['args'].get('project', 'default_project')
        name    = cfg['logger']['args'].get('name', 'scanrefer')
        
        # Initialize a Wandb run
        self.run = wandb.init(
            entity=entity,
            project=project,
            dir=log_dir,
            config=cfg.get('config', {}),
            name=name
        )
    
    def log(self, log_dict, step=None):
        if step is not None:
            log_dict['step'] = step
        self.run.log(log_dict)

    def __del__(self):
        self.run.finish()