from ultralytics import YOLO
from pathlib import Path
import argparse
import shutil
import os

custom_cfg = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'translate': 0.1,
    'flipud': 0.5,  # default = 0.0
    'fliplr': 0.5,
    'degrees': 0.0,  # default = 0.0
    'scale': 0.0,  # default = 0.5
    'shear': 0.0,  # default = 0.0
    'perspective': 0.0,  # default = 0.0
    'mosaic': 0.0,  # default = 1.0
    'mixup': 0.0,  # default = 0.0
    'copy_paste': 0.0,  # default = 0.0
}

# Building the model
# model = YOLO(f"./YOLO_config_files/{model_name}.yaml", overrides=custom_cfg)  # load an untrained YOLO model
model = YOLO()  # load an untrained YOLO model
model.overrides = custom_cfg


# add custom arguments into YOLO model wrt augmentation params


# Training the model
model.train(
    data="./data-plate-border.yaml",
    epochs=10,
    box=box_gain,
    cls=class_gain,
    dfl=dfl_gain,
    imgsz=(1280, 1600),
    optimizer=args.optimizer
)

# Validating the model
res = model.val(data="./data-plate-border.yaml")

# Make results available in Azure ML Studio
shutil.copytree("runs", "./outputs/runs")


