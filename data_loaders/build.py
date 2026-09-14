"""Build transforms, datasets, and data loaders from a configuration."""

from torch.utils.data import DataLoader

from utils import TRANSFORMS

from .dataset import Dataset
from .transforms import Compose


def build_transforms(cfg, mode='train'):
    """Transforms for train or test."""
    pipeline_cfg = cfg.TRAIN_PIPELINE if mode == 'train' else cfg.TEST_PIPELINE

    pipeline = []
    for transform_cfg in pipeline_cfg:
        cfg_copy = dict(transform_cfg)
        t_type = cfg_copy.pop('type')
        t_cls = TRANSFORMS.get(t_type)
        if t_cls is None:
            raise KeyError(f"{t_type} is not registered in TRANSFORMS!")
        pipeline.append(t_cls(**cfg_copy))
    # print('[debug] transforms', mode, [type(t).__name__ for t in pipeline])
    return Compose(pipeline)


def build_dataset(cfg, mode='train'):
    """Build the Dataset for one split using paths from cfg.{MODE}_*."""
    transforms = build_transforms(cfg, mode=mode)

    mode_upper = mode.upper()
    ann_file       = getattr(cfg, f"{mode_upper}_ANN_FILE")
    img_dir        = getattr(cfg, f"{mode_upper}_IMG_DIR")
    # Validation/test group labels.
    occlusion_file = getattr(cfg, f"{mode_upper}_OCCLUSION_FILE", None)
    not_occluded_file = getattr(
        cfg, f"{mode_upper}_NOT_OCCLUDED_FILE", None
    )

    # Mask centres for centroid experiments.
    cm_cfg = (getattr(cfg, "MODEL", {}).get("head", {}).get("centroid_module") or {})
    compute_inner_centers = bool(cm_cfg.get("enabled", False))

    return Dataset(
        parser_type           = cfg.PARSER_TYPE,
        ann_file              = ann_file,
        img_dir               = img_dir,
        transforms            = transforms,
        prefix_filter         = getattr(cfg, "FILE_PREFIX", None),
        occlusion_label_file  = occlusion_file,
        not_occluded_label_file = not_occluded_file,
        category_remap        = getattr(cfg, "CATEGORY_REMAP", None),
        compute_inner_centers = compute_inner_centers,
    )


def detection_collate_fn(batch):
    """Image list and matching target list.

    Each image: Tensor[3, H_i, W_i]. Resizing/padding in the model transform.
    """
    images, targets = zip(*batch)
    # print("[debug] batch", len(images), [im.shape for im in images])
    return list(images), list(targets)


def build_dataloader(dataset, cfg, mode='train'):
    """Loader for one split."""
    # print('[debug]', mode, 'samples', len(dataset), 'workers', cfg.NUM_WORKERS)
    return DataLoader(
        dataset,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = (mode == 'train'),
        num_workers = cfg.NUM_WORKERS,
        collate_fn  = detection_collate_fn,
    )
