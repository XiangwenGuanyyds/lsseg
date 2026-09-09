"""Parse COCO annotations into the records used by the data pipeline.

The parser supports filename filtering, category remapping, and conversion
from COCO ``xywh`` boxes to ``xyxy`` boxes. It excludes crowd annotations and
degenerate boxes from model targets.
"""

import json
import os

from utils import PARSERS
from .base_parser import BaseParser


@PARSERS.register_module()
class COCOParser(BaseParser):
    """Parse a COCO-format annotation JSON.

    Args:
        ann_file:        Path to the COCO JSON.
        img_dir:         Directory containing image files.
        prefix_filter:   Optional filename prefix.
        category_remap:  Optional mapping from source to training category IDs.
    """

    def __init__(self, ann_file, img_dir, prefix_filter=None,
                 category_remap=None):
        super().__init__(ann_file)
        self.img_dir = img_dir
        self.prefix_filter = prefix_filter
        self.category_remap = dict(category_remap) if category_remap else None
        self.categories = []

    def load_data(self):
        """Read a COCO JSON file and return one record per retained image.

        Returns:
            list[dict], one per kept image:
                {
                    "id":          int,    # COCO image id
                    "file_path":   str,    # image path
                    "height":      int,    # image height
                    "width":       int,    # image width
                    "annotations": [
                        {
                            "id":           int,             # COCO annotation id
                            "category_id":  int,             # remapped category ID
                            "segmentation": list,            # COCO polygon list
                            "bbox":         [x1,y1,x2,y2],   # converted from xywh
                        },
                        ...
                    ]
                }

            ``self.categories`` stores the retained category definitions.
        """
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        self.categories = self._build_category_list(coco_data.get('categories', []))
        filtered_images = self._filter_images(coco_data['images'])
        per_image_anns, n_dropped_cat = self._group_annotations(
            coco_data['annotations'], set(filtered_images.keys())
        )

        # Keep images that contain at least one valid annotation.
        final_list = []
        for img_id, img_info in filtered_images.items():
            anns = per_image_anns.get(img_id, [])
            if not anns:
                continue
            final_list.append({
                "id":          img_id,
                "file_path":   os.path.join(self.img_dir, img_info['file_name']),
                "height":      img_info['height'],
                "width":       img_info['width'],
                "annotations": anns,
            })

        msg = (f"[Parser] Filtered by '{self.prefix_filter}': "
               f"Kept {len(final_list)}/{len(coco_data['images'])} images.")
        if self.category_remap is not None:
            msg += (f" Category remap {self.category_remap} dropped "
                    f"{n_dropped_cat} annotation(s).")
        print(msg)
        # print(f"[COCOParser] categories={self.categories}, records={len(final_list)}, annotations={sum(len(item['annotations']) for item in final_list)}")
        return final_list

    def _build_category_list(self, all_categories):
        """Apply ``category_remap`` to the COCO categories array."""
        if self.category_remap is None:
            return all_categories
        id_to_cat = {c['id']: c for c in all_categories}
        new_cats = []
        for src_id, dst_id in self.category_remap.items():
            cat = id_to_cat.get(src_id)
            if cat is None:
                continue
            new_cat = dict(cat)
            new_cat['id'] = dst_id
            new_cats.append(new_cat)
        return new_cats

    def _filter_images(self, images):
        """Return ``{image_id: image_dict}`` for images passing ``prefix_filter``."""
        kept = {}
        for img in images:
            if self.prefix_filter and not img['file_name'].startswith(self.prefix_filter):
                continue
            kept[img['id']] = img
        return kept

    def _group_annotations(self, annotations, valid_img_ids):
        """Bucket annotations by image_id, apply remap, drop degenerate bboxes.

        Returns ``(per_image_anns, n_dropped_cat)``.
        """
        per_image_anns = {}
        n_dropped_cat = 0

        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in valid_img_ids:
                continue
            if int(ann.get('iscrowd', 0)) != 0:
                continue

            src_cat = ann['category_id']
            if self.category_remap is not None:
                if src_cat not in self.category_remap:
                    n_dropped_cat += 1
                    continue
                dst_cat = self.category_remap[src_cat]
            else:
                dst_cat = src_cat

            pascal_bbox = self._coco_bbox_to_pascal(ann['bbox'])
            if pascal_bbox[2] <= pascal_bbox[0] or pascal_bbox[3] <= pascal_bbox[1]:
                continue

            per_image_anns.setdefault(img_id, []).append({
                "id":           ann['id'],
                "category_id":  dst_cat,
                "segmentation": ann['segmentation'],
                "bbox":         pascal_bbox,
            })

        return per_image_anns, n_dropped_cat

    @staticmethod
    def _coco_bbox_to_pascal(bbox):
        """COCO ``[x, y, w, h]`` -> Pascal ``[x1, y1, x2, y2]``."""
        converted = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        # print(f"[COCOParser] bbox_xywh={bbox}, bbox_xyxy={converted}")
        return converted
