#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/train_resnet.py \
  --snapshot-path "$FOLDERNAME/snapshots" \
  --tensorboard-dir "$FOLDERNAME/logs" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --epochs 1 \
  --steps 1 \
  --tensorboard-update-freq 20 \
    csv \
  --base_dir "$DATASET_FOLDERNAME/images_patches" \
  --annotations "$DATASET_FOLDERNAME/annotations/patches/patches_annotations_train_157_lines.csv" \
  --val-annotations "$DATASET_FOLDERNAME/annotations/patches/patches_annotations_val_157_lines.csv"
  # --max-annotations 157 \