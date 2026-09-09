"""scene_pig + 原装 torchvision Mask R-CNN（对照基线）。

跟 scene_pig_baseline 的唯一区别：MODEL 用 NativeMaskRCNN，整个链路是
torchvision 官方 Mask R-CNN（transform 内部做、backbone 内部做、head
内部做），用来对照 Architecture 的拆解版本是否还原原装行为。

Run:
    python main.py --mode train --config scene_pig_native --exp_name <name>
"""

import os

import torch

from utils.registry import CONFIGS


@CONFIGS.register_module(name="scene_pig_native")
class ScenePigNativeConfig:

    # ============================================================
    # 通用
    # ============================================================
    TASK_NAME   = "scene_pig_native"
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    SEED        = 42
    NUM_EPOCHS  = 20
    BATCH_SIZE  = 4
    NUM_WORKERS = 2

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    DEBUG = {
        "mask_ap_input": False,   # True → 进 COCOeval 前打印 GT/预测形状
    }

    # ============================================================
    # 数据集
    # ============================================================
    PARSER_TYPE    = "COCOParser"
    CATEGORY_REMAP = {2: 1}

    PER_SCENE_DIR = os.path.join(BASE_DIR, "dataset", "per_scene", "pig")
    RAW_DIR       = os.path.join(BASE_DIR, "dataset", "raw", "pigs", "pig")

    TRAIN_IMG_DIR  = os.path.join(RAW_DIR, "train")
    TRAIN_ANN_FILE = os.path.join(PER_SCENE_DIR, "train.coco.json")
    VAL_IMG_DIR    = os.path.join(RAW_DIR, "valid")
    VAL_ANN_FILE   = os.path.join(PER_SCENE_DIR, "val.coco.json")
    TEST_IMG_DIR   = os.path.join(RAW_DIR, "test")
    TEST_ANN_FILE  = os.path.join(PER_SCENE_DIR, "test.coco.json")

    TRAIN_PIPELINE = [
        {"type": "RandomHorizontalFlip"},
        {"type": "ToTensor"},
    ]
    TEST_PIPELINE = [
        {"type": "ToTensor"},
    ]

    # ============================================================
    # 模型
    # ============================================================
    MODEL = {
        "type": "NativeMaskRCNN",
        "num_classes": 2,
        "full_maskrcnn_weights": None,
        "resnet50_body_weights": "IMAGENET1K_V2",
        "trainable_resnet_layers": 3,
    }
    LOSS      = {"type": "LossAggregator"}
    OPTIMIZER = {"type": "SGD", "lr": 0.0002}

    # ============================================================
    # 评估
    # ============================================================
    # NativeMaskRCNN 不暴露 self.state，无诊断 hook，所以这里只配 EVAL_CLASS_IDS。
    EVAL_CLASS_IDS = [1]
