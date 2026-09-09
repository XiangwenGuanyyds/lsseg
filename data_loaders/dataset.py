"""Load images and build Mask R-CNN target tensors."""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset as torchDataset

from utils import PARSERS


def _pole_of_inaccessibility(mask_np):
    """Find the mask pixel furthest from its boundary.

    The distance transform uses the cropped mask with a one-pixel zero border.
    Returns ``(cx, cy)`` or ``None`` for an empty mask.
    """
    if mask_np.sum() == 0:
        return None

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    cropped = mask_np[rmin:rmax + 1, cmin:cmax + 1].astype(np.uint8)

    padded = np.zeros((cropped.shape[0] + 2, cropped.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = cropped

    d = distance_transform_edt(padded)
    cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(cx - 1 + cmin), float(cy - 1 + rmin)


class Dataset(torchDataset):
    """Load images and instance annotations for model input.

    Args:
        parser_type: Registered parser name, such as ``COCOParser``.
        ann_file: Path to the annotation JSON file.
        img_dir: Directory containing the images.
        transforms: Optional transform pipeline for the image and target.
        prefix_filter: Optional filename prefix used to select images.
        occlusion_label_file: Optional JSON list of occluded annotation IDs.
        not_occluded_label_file: Optional JSON list of not-occluded annotation
            IDs. An omitted file assigns group labels using the complement of
            the occluded ID list.
        category_remap: Mapping from source category IDs to training IDs.
        compute_inner_centers: Compute one ``inner_centers`` value per mask.
    """

    def __init__(self,
                 parser_type,
                 ann_file,
                 img_dir,
                 transforms=None,
                 prefix_filter=None,
                 occlusion_label_file=None,
                 not_occluded_label_file=None,
                 category_remap=None,
                 compute_inner_centers=False):

        parser_cls = PARSERS.get(parser_type)
        if parser_cls is None:
            raise KeyError(f"Parser '{parser_type}' not found in PARSERS registry.")

        self.parser = parser_cls(ann_file, img_dir,
                                 prefix_filter=prefix_filter,
                                 category_remap=category_remap)
        self.category_remap = self.parser.category_remap
        self.data_infos = self.parser.load_data()
        self.categories = getattr(self.parser, 'categories', [])

        self.transforms = transforms
        self.occlusion_label_file = occlusion_label_file
        self.not_occluded_label_file = not_occluded_label_file
        self._occluded_ids = self._load_label_ids(
            occlusion_label_file, "occluded"
        )
        self._not_occluded_ids = (
            self._load_label_ids(not_occluded_label_file, "not_occluded")
            if not_occluded_label_file is not None
            else None
        )
        if self._not_occluded_ids is not None:
            overlap = self._occluded_ids & self._not_occluded_ids
            if overlap:
                raise ValueError(
                    "Occluded and not-occluded label files overlap for "
                    f"{len(overlap)} annotation ids"
                )
        self._compute_inner_centers = bool(compute_inner_centers)

    @staticmethod
    def _load_label_ids(path, label_name):
        """Load annotation IDs for one occlusion group from a JSON list."""
        if not path:
            return set()
        if not os.path.exists(path):
            print(f"  [Dataset] {label_name} labels not found at {path}.")
            return set()

        with open(path) as f:
            raw = json.load(f)

        ids = {int(x) for x in raw}
        print(f"  [Dataset] Loaded {len(ids)} {label_name} ann_ids from {path}.")
        return ids

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        """Load one image and construct its target dictionary.

        Returns:
            A tuple containing ``image`` and ``target``. ``image`` is a PIL
            image or a ``Tensor[3, H, W]`` after ``ToTensor``. ``target``
            contains ``N`` ground-truth instances:

                boxes         Tensor[N, 4]    float32  xyxy coordinates
                labels        Tensor[N]       int64    remapped category IDs
                masks         Tensor[N, H, W] uint8    binary instance masks
                is_occluded   Tensor[N]       bool     occlusion labels
                has_occlusion_label Tensor[N] bool     valid group labels
                ann_ids       Tensor[N]       int64    COCO annotation IDs
                image_id      Tensor[1]       int64    COCO image ID
                orig_size     Tensor[2]       int64    original (height, width)
                inner_centers Tensor[N, 2]    float32  optional mask centres
        """
        info = self.data_infos[index]
        real_img_id = info.get('id')
        image = Image.open(info['file_path']).convert("RGB")
        orig_h, orig_w = info['height'], info['width']
        anns = info['annotations']
        # print(f"[Dataset] index={index}, image_size={image.size}, annotations={len(anns)}")

        masks = [self._poly_to_mask(a['segmentation'], orig_h, orig_w) for a in anns]
        is_occluded = [a['id'] in self._occluded_ids for a in anns]
        if self._not_occluded_ids is None:
            has_occlusion_label = [True] * len(anns)
        else:
            labeled_ids = self._occluded_ids | self._not_occluded_ids
            has_occlusion_label = [a['id'] in labeled_ids for a in anns]

        target = {
            "boxes":       torch.as_tensor([a['bbox'] for a in anns], dtype=torch.float32),
            "labels":      torch.as_tensor([a['category_id'] for a in anns], dtype=torch.int64),
            "masks":       torch.as_tensor(np.array(masks), dtype=torch.uint8),
            "is_occluded": torch.as_tensor(is_occluded, dtype=torch.bool),
            "has_occlusion_label": torch.as_tensor(
                has_occlusion_label, dtype=torch.bool
            ),
            "ann_ids":     torch.as_tensor([a['id'] for a in anns], dtype=torch.int64),
            "image_id":    torch.tensor([real_img_id]),
            "orig_size":   torch.tensor([orig_h, orig_w]),
        }

        if self._compute_inner_centers:
            ic_list = []
            for m in masks:
                p = _pole_of_inaccessibility(m)
                ic_list.append([p[0], p[1]] if p is not None else [-1.0, -1.0])
            target["inner_centers"] = (
                torch.as_tensor(ic_list, dtype=torch.float32) if ic_list
                else torch.zeros((0, 2), dtype=torch.float32)
            )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # print(f"[Dataset] image_shape={tuple(image.shape)}, boxes={tuple(target['boxes'].shape)}, masks={tuple(target['masks'].shape)}, labels={tuple(target['labels'].shape)}")

        return image, target

    @staticmethod
    def _poly_to_mask(polygons, h, w):
        """Rasterise one COCO polygon annotation into an ``H x W`` mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            p = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [p], color=1)  # type: ignore
        # print(f"[Dataset] polygon_count={len(polygons)}, mask_shape={mask.shape}, foreground_pixels={int(mask.sum())}")
        return mask
