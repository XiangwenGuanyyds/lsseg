import torch.optim as optim


class Registry:
    """Decorator-based registry. Components self-register at import time.

    Usage:
        @MODELS.register_module()                    # uses class.__name__
        class ResNet50FPN(...): ...

        @CONFIGS.register_module(name='pig_baseline_100ep')   # custom name
        class PigBaseline100EpConfig(...): ...

    Look up:
        cls = MODELS.get('ResNet50FPN')
        cls = CONFIGS.get('pig_baseline_100ep')
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self, name=None):
        """Decorator. If ``name`` is given, register under that key;
        otherwise use ``cls.__name__`` (backward-compatible)."""

        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(
                    f"{self._name} registry: '{key}' already registered "
                    f"(by {self._module_dict[key].__name__})."
                )
            self._module_dict[key] = cls
            return cls

        return _register

    def get(self, key):
        return self._module_dict.get(key)

    def keys(self):
        return sorted(self._module_dict.keys())

    def __contains__(self, key):
        return key in self._module_dict

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        return f"Registry('{self._name}', {len(self)} entries)"


PARSERS = Registry('parsers')
MODELS = Registry('models')
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
MASK_LOSSES = Registry("mask_losses")
BOX_NMS = Registry("box_nms")
BOX_REG_LOSSES = Registry("box_reg_losses")
CONFIGS = Registry('configs')

OPTIMIZERS = Registry("optimizers")
OPTIMIZERS._module_dict.update({
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "RMSprop": optim.RMSprop
})
