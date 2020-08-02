#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/predict_resnet.py \
  --base_dir "$DATASET_FOLDERNAME/images_patches" \
  --out "$DATASET_FOLDERNAME" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --max-annotations 157 \
  --save-predicted-images \
    csv \
  --annotations "$DATASET_FOLDERNAME/annotations/patches/patches_annotations_val_157_lines.csv" \
  "$FOLDERNAME/snapshots/Sat_Aug__1_19_28_59_2020/resnet50_csv_01.h5"