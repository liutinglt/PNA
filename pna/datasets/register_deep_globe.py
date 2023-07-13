# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
 

def load_deep_globe_semantic(root_dir, list_path):
    files = []
    # scan through the directory 
    ret = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            image_file, label_file = line.strip().split("\t")
            ret.append(
            {
                "file_name": os.path.join(root_dir, image_file),
                "sem_seg_file_name": os.path.join(root_dir,label_file),
                "height": 2448,
                "width": 2448,
            }
            )
    assert len(ret), f"No images found in {root_dir}!"  
    return ret      
 
_RAW_GLOBE_SPLITS = {
    "deep_globe_train": ("DeepGlobe", "DeepGlobe/train.txt"),
    "deep_globe_val": ("DeepGlobe", "DeepGlobe/test.txt"),
    "deep_globe_test": ("DeepGlobe", "DeepGlobe/test.txt"),
}

def register_all_deep_globe(root):
    class_colors = {'unknown': [0, 0, 0],
           'unknown': [0, 255, 255],
            'agriculture_land': [255, 255, 0],
            'rangeland': [255, 0, 255],
            'forest_land': [0, 255, 0],
            'water': [0, 0, 255],
            'barren_land': [255, 255, 255]}
    classes = [ k for k, v in class_colors.items()]
    colors = [v for k, v in class_colors.items()] 
    for key, (file_dir, list_file) in _RAW_GLOBE_SPLITS.items():
        root_dir = os.path.join(root, file_dir)
        list_path = os.path.join(root, list_file)
        name = key
        DatasetCatalog.register(
            name, lambda x=root_dir, y=list_path: load_deep_globe_semantic(x, y)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=classes,
            stuff_colors=colors,  
            image_root=root_dir,
            sem_seg_root=root_dir,
            evaluator_type="sem_seg", 
            ignore_label=255,
        )
   
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_deep_globe(_root)