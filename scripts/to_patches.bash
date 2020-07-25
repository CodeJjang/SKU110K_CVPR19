#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/utils/to_patches.py \
  --images "$DATASET_FOLDERNAME/images" \
  --images-out "$DATASET_FOLDERNAME/images_patches" \
  --csv-out "$DATASET_FOLDERNAME/annotations/patches" \
  --train-annotations "$DATASET_FOLDERNAME/annotations/annotations_train_157_lines.csv" \
  --val-annotations "$DATASET_FOLDERNAME/annotations/annotations_val_157_lines.csv" \
  --test-annotations "$DATASET_FOLDERNAME/annotations/annotations_test.csv"