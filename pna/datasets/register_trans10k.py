# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg 
 
def register_all_trans10k(root):
    root = os.path.join(root, "trans10k")
    for name, dirname in [("train", "train"), ("test", "test"), ("test_hard", "test_hard"), ("test_easy", "test_easy"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"trans10k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=["background", "things", "stuff"],
            image_root=image_dir,
            stuff_colors=[[0, 0, 0], [255, 0, 0], [255, 255, 255]],
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_trans10k(_root)