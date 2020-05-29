#!/bin/bash
python -u object_detector_retinanet/keras_retinanet/bin/train.py \
  --snapshot-path "$FOLDERNAME/snapshots" \
  --tensorboard-dir "$FOLDERNAME/logs" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --epochs 1 \
  --max-annotations 157 \
    csv \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_train.csv" \
  --val-annotations "$DATASET_FOLDERNAME/annotations/annotations_val.csv"