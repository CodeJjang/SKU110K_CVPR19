#!/bin/bash
python -u object_detector_retinanet/keras_retinanet/bin/predict.py \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --out "$DATASET_FOLDERNAME" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
    csv \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_val_157_ann.csv" \
  "$DATASET_FOLDERNAME/iou_resnet50_csv_06.h5"