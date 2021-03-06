#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/predict.py \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --out "$DATASET_FOLDERNAME" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --max-annotations 157 \
    csv \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_val.csv" \
  --vanilla-retinanet \
  --max-detections 1 \
  --iou-threshold 0.5 \
  "$DATASET_FOLDERNAME/iou_resnet50_csv_06.h5"