#!/bin/bash
python -u object_detector_retinanet/keras_retinanet/utils/map_over_hard_score_rate.py \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --out "$DATASET_FOLDERNAME" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --max-annotations 157 \
    csv \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_val.csv" \
  "$DATASET_FOLDERNAME/iou_resnet50_csv_06.h5"