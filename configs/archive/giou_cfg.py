"""Mixed-scene + GIoU box-regression loss.

Same as baseline except cfg.MODEL["box_reg_loss"] = "giou".

Box 头回归损失从 SmoothL1（torchvision 默认，β=1/9）替换为 GIoU loss
(Rezatofighi et al. 2019, CVPR — "Generalized Intersection over Union")。
RPN / box 头分类（仍是 CE）/ NMS / mask 头 / 训练流水**完全不动**。

Run:
    python main.py --mode train --config giou --exp_name <name>
    python main.py --mode test  --config giou --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="giou")
class GIoUConfig(BaselineConfig):
    TASK_NAME = "giou"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_reg_loss": "giou",
    }
