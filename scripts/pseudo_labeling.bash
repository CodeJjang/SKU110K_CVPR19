#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/utils/pseudo_labeling.py \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_train_0.csv" \
  --predicted-annotations "$DATASET_FOLDERNAME/results/fake_train_det.csv" \
  --out "./" \
  --dup-removal-tactic em-merger