#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/train.py \
  --snapshot-path "$FOLDERNAME/bay-snapshots" \
  --tensorboard-dir "$FOLDERNAME/bay-logs" \
  --images-cls-cache "$FOLDERNAME/bay-images_cls_cache" \
  --epochs 1 \
  --steps 1 \
  --tensorboard-update-freq 20 \
  --max-annotations 157 \
  --snapshots \
  --save-freq 1 \
    csv \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --annotations "$DATASET_FOLDERNAME/annotations/bay/bay_annotations_train.csv" \
  --val-annotations "$DATASET_FOLDERNAME/annotations/bay/bay_annotations_val.csv" 