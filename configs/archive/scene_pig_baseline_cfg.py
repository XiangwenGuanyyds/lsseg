"""Single-scene baseline: pig scene + standard Mask R-CNN.

Run:
    python main.py --mode train --config scene_pig_baseline --exp_name <name>
    python main.py --mode train --config scene_pig_baseline --exp_name <name> --seed 1
"""

import os

import torch

from utils.registry import CONFIGS


@CONFIGS.register_module(name="scene_pig_baseline")
class ScenePigBaselineConfig:

    # ============================================================
    # 通用
    # ============================================================
    TASK_NAME   = "scene_pig_baseline"
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    SEED        = 42
    NUM_EPOCHS  = 20
    BATCH_SIZE  = 4
    NUM_WORKERS = 2

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Debug print switches — each True turns on a single trace point.
    DEBUG = {
        "fpn_input":         False,
        "fpn_output":        False,
        "roi_align_output":  False,   # True → 打印 box 分支 RoIAlign 输出形状
        "box_output":        False,   # True → 打印 box_head 出 class_logits / box_regression 形状
        "mask_input":        False,   # True → 打印 mask_proposals per-image 数 + mask_roi_pool 输出形状
        "mask_ap_input":     False,   # True → 进 COCOeval 前打印 GT/预测形状
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

    # Per-split occ files (annotator writes val + test only). train 没标，
    # 故不定义 TRAIN_OCCLUSION_FILE；缺文件即该 split 所有实例默认 not_occluded、
    # mask_ap 也只算 overall。
    VAL_OCCLUSION_FILE  = os.path.join(PER_SCENE_DIR, "occ_ann.val.json")
    TEST_OCCLUSION_FILE = os.path.join(PER_SCENE_DIR, "occ_ann.test.json")

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
        "type": "Architecture",
        "full_maskrcnn_weights": None,
        "resnet50_body_weights": "IMAGENET1K_V2",
        "trainable_resnet_layers": 3,
        "head":      {"num_classes": 2},
    }
    LOSS      = {"type": "LossAggregator"}
    OPTIMIZER = {"type": "SGD", "lr": 0.0002}

    # ============================================================
    # 评估 / 诊断
    # ============================================================
    EVAL_CLASS_IDS = [1]                      # post-CATEGORY_REMAP class id (pig)
    DIAGNOSTICS    = {                        # test stage writes outputs/<experiment>/test and analysis outputs
        "rpn_coverage": True,
    }
