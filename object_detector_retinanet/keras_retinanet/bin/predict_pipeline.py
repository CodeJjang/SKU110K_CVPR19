#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import keras
import numpy
import tensorflow as tf
import logging

from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from object_detector_retinanet.keras_retinanet.utils.predict_pipeline import predict as predict_pipeline
from object_detector_retinanet.keras_retinanet.utils.predict import predict as predict_vanilla
from object_detector_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.utils import image_path, annotation_path, root_dir
from object_detector_retinanet.keras_retinanet.utils.to_coco import print_metrics


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_objects_generator(args):
    """ Create objects generator for evaluation.
    """
    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.objects_annotations,
            args.objects_classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            base_dir=args.base_dir,
            images_cls_cache_path=args.objects_images_cls_cache
        )
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def create_bays_generator(args):
    """ Create bays generator for evaluation.
    """
    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.bays_annotations,
            args.bays_classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            base_dir=args.base_dir,
            images_cls_cache_path=args.bays_images_cls_cache
        )
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluation script for the pipeline.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument(
        'coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument(
        'pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    args_images_cls_cache = os.path.join(root_dir(), 'images_cls_cache')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument(
        '--objects-annotations', help='Path to CSV file containing annotations for objects evaluation.')
    csv_parser.add_argument(
        '--bays-annotations', help='Path to CSV file containing annotations for bays evaluation.')
    csv_parser.add_argument('--objects_classes', help='Path to a CSV file containing class label mapping.',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_mappings.csv'))
    csv_parser.add_argument('--bays_classes', help='Path to a CSV file containing class label mapping.',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bays_class_mappings.csv'))
    parser.add_argument('--hard_score_rate', help='', default="0.5")

    parser.add_argument('--object-detection-model',
                        help='Path to RetinaNet object detection model.')
    parser.add_argument('--bay-detection-model',
                        help='Path to RetinaNet bay detection model.')
    parser.add_argument('--base_dir', help='Path to base dir for images file.',
                        default=image_path())
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).', type=int,
                        default=1)

    parser.add_argument(
        '--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument(
        '--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.1, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.5, type=float)
    parser.add_argument(
        '--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)
    parser.add_argument('--objects-images-cls-cache',
                        help='Path to store images objects classes cache (for faster loading when images are stored in the cloud)',
                        default=args_images_cls_cache)
    parser.add_argument('--bays-images-cls-cache',
                        help='Path to store images bays classes cache (for faster loading when images are stored in the cloud)',
                        default=args_images_cls_cache)
    parser.add_argument('--out', help='Path to out dir results.')
    parser.add_argument(
        '--max-annotations', help='Trim annotations to max number (easier debugging)', type=int)
    parser.add_argument(
        '--flush-csv-freq', help='Frequency of images of flushing detections to csv', type=int)
    parser.add_argument('--res-file-path',
                        help='Path of previously saved detections csv')
    parser.add_argument('--save-predicted-images',
                        help='Whether to save predicted images with boxes (slows down inference)', action='store_true')
    parser.add_argument(
        '--max-detections', help='Maximum number of detections', type=int, default=9999)
    return parser.parse_args(args)


def main(args=None):
    configure_logging()
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.hard_score_rate:
        hard_score_rate = float(args.hard_score_rate.lower())
    else:
        hard_score_rate = 0.5
    logging.info("hard_score_rate={}".format(hard_score_rate))

    args.max_annotations = args.max_annotations if args.max_annotations is not None and args.max_annotations > 0 else None

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    use_cpu = False

    if args.gpu:
        gpu_num = args.gpu
    else:
        gpu_num = str(0)

    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(666)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    objects_generator = create_objects_generator(args)
    bays_generator = create_bays_generator(args)

    # load the model
    logging.info('Loading models, this may take a second...')
    bay_detection_model = models.load_model(
        args.bay_detection_model, backbone_name=args.backbone, convert=args.convert_model, nms=False)
    object_detection_model = models.load_model(
        args.object_detection_model, backbone_name=args.backbone, convert=args.convert_model, nms=False)

    save_predicted_images_path = None
    if args.save_predicted_images:
        save_predicted_images_path = os.path.join(args.out, 'res_images_iou')

    # Predict on pipeline
    dt_annotations_path = predict_pipeline(
        objects_generator,
        bays_generator,
        bay_detection_model,
        object_detection_model,
        max_detections=args.max_detections,
        score_threshold=args.score_threshold,
        save_path=save_predicted_images_path,
        hard_score_rate=hard_score_rate,
        base_dir=args.base_dir,
        out_dir=args.out,
        flush_csv_freq=args.flush_csv_freq,
        res_file=args.res_file_path,
        max_annotations=args.max_annotations
    )
    # Print metrics
    return print_metrics(args.objects_annotations, dt_annotations_path, args.max_annotations)


if __name__ == '__main__':
    main()
