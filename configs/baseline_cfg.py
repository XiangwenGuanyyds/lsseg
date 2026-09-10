"""Standard FPN + Mask R-CNN baseline configuration."""

import os

import torch

from utils.registry import CONFIGS


@CONFIGS.register_module(name="baseline")
class BaselineConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    NUM_EPOCHS = 50
    BATCH_SIZE = 2
    NUM_WORKERS = 2

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DEBUG = {
        "fpn_input": False,
        "fpn_output": False,
        "roi_align_output": False,
        "box_output": False,
        "mask_input": False,
        "mask_ap_input": False,
    }

    PARSER_TYPE = "COCOParser"
    CATEGORY_REMAP = {2: 1}

    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "per_scene", "mix")
    RAW_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "raw", "pigs", "pig")

    TRAIN_IMG_DIR = os.path.join(RAW_IMAGE_DIR, "train")
    TRAIN_ANN_FILE = os.path.join(DATASET_DIR, "train.coco.json")
    VAL_IMG_DIR = os.path.join(RAW_IMAGE_DIR, "valid")
    VAL_ANN_FILE = os.path.join(DATASET_DIR, "val.coco.json")
    TEST_IMG_DIR = os.path.join(RAW_IMAGE_DIR, "test")
    TEST_ANN_FILE = os.path.join(DATASET_DIR, "test.coco.json")

    CONSENSUS_DIR = os.path.join(DATASET_DIR, "occlusion_review", "consensus")
    VAL_OCCLUSION_FILE = os.path.join(CONSENSUS_DIR, "occluded.val.json")
    VAL_NOT_OCCLUDED_FILE = os.path.join(
        CONSENSUS_DIR, "not_occluded.val.json"
    )
    TEST_OCCLUSION_FILE = os.path.join(CONSENSUS_DIR, "occluded.test.json")
    TEST_NOT_OCCLUDED_FILE = os.path.join(
        CONSENSUS_DIR, "not_occluded.test.json"
    )

    TRAIN_PIPELINE = [
        {"type": "RandomHorizontalFlip"},
        {"type": "ToTensor"},
    ]
    TEST_PIPELINE = [{"type": "ToTensor"}]

    MODEL = {
        "type": "MaskRCNN",
        "full_maskrcnn_weights": None,
        "resnet50_body_weights": "IMAGENET1K_V2",
        "trainable_resnet_layers": 3,
        "head": {"num_classes": 2},
        "mask_loss": "maskrcnn_loss",
    }
    LOSS = {"type": "LossAggregator"}
    OPTIMIZER = {"type": "SGD", "lr": 0.0002}

    EVAL_CLASS_IDS = [1]
    DIAGNOSTICS = {
        "rpn_coverage": True,
        "postprocessing_recall": True,
    }
