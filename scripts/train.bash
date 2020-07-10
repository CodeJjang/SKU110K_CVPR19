#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/train.py \
  --snapshot-path "$FOLDERNAME/snapshots" \
  --tensorboard-dir "$FOLDERNAME/logs" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --epochs 1 \
  --steps 1 \
  --tensorboard-update-freq 20 \
  --max-annotations 157 \
    csv \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --annotations "$DATASET_FOLDERNAME/annotations/pseudo-label-nms/detections_output_iou_0.5_2020-07-08 22_57_23.295040_0.3_thres.csv" \
  --val-annotations "$DATASET_FOLDERNAME/annotations/annotations_val.csv"