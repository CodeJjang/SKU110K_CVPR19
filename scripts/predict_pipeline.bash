#!/bin/bash
FLAGS="-u"
if [ "$DEBUG" = "true" ]; then
  FLAGS="-m debugpy --listen 5678 --wait-for-client"
fi
python $FLAGS object_detector_retinanet/keras_retinanet/bin/predict_pipeline.py \
  --base_dir "$DATASET_FOLDERNAME/images" \
  --out "$DATASET_FOLDERNAME" \
  --images-cls-cache "$FOLDERNAME/images_cls_cache" \
  --max-annotations 157 \
  --object-detection-model "$DATASET_FOLDERNAME/iou_resnet50_csv_06.h5" \
  --bay-detection-model "$DATASET_FOLDERNAME/snapshots-bay-detector/Sun_Jul_19_14_34_31_2020/resnet50_csv_03.h5" \
    csv \
  --annotations "$DATASET_FOLDERNAME/annotations/annotations_val.csv"