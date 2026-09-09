import torch
from utils import OPTIMIZERS


def build_optimizer(cfg, model):


    opt_cfg = cfg.OPTIMIZER.copy()


    optimizer_type = opt_cfg.pop('type')

    optimizer_cls = OPTIMIZERS.get(optimizer_type)

    if optimizer_cls is None:
        raise KeyError(f"Optimizer '{optimizer_type}' is not registered!")

    # Optional: per-parameter-group LR. When OPTIMIZER["param_groups"] is not
    # present (every existing config), behavior is bit-identical to the old
    # single-group path below.
    param_groups_cfg = opt_cfg.pop('param_groups', None)

    if param_groups_cfg is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return optimizer_cls(params, **opt_cfg)

    base_lr = opt_cfg['lr']
    scratch_mult = param_groups_cfg['scratch_lr_mult']
    patterns = param_groups_cfg['scratch_param_patterns']

    pretrained_params, scratch_params = [], []
    pretrained_names, scratch_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(pat in name for pat in patterns):
            scratch_params.append(p)
            scratch_names.append(name)
        else:
            pretrained_params.append(p)
            pretrained_names.append(name)

    if not scratch_params:
        raise ValueError(
            f"param_groups configured with patterns {patterns} but no "
            f"matching parameter found in model. Check the patterns against "
            f"model.named_parameters()."
        )

    print(f"[build_optimizer] param-group split:")
    print(f"  pretrained group: {len(pretrained_params)} tensors @ lr={base_lr}")
    print(f"  scratch    group: {len(scratch_params)} tensors @ lr={base_lr * scratch_mult}")
    print(f"  scratch param names: {scratch_names}")

    groups = [
        {'params': pretrained_params, 'lr': base_lr},
        {'params': scratch_params,    'lr': base_lr * scratch_mult},
    ]
    return optimizer_cls(groups, **opt_cfg)