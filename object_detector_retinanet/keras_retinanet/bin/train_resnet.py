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
import time
import logging

import keras
import keras.preprocessing.image
import tensorflow as tf
import keras_resnet
import keras_resnet.models
# Allow relative imports when being executed as script.

# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#     __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
# from .. import losses
from object_detector_retinanet.keras_retinanet import losses
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.callbacks import RedirectModel
from object_detector_retinanet.keras_retinanet.models.retinanet import retinanet_bbox
from object_detector_retinanet.keras_retinanet.preprocessing.csv_classifier_generator import CSVClassifierGenerator
from object_detector_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from object_detector_retinanet.keras_retinanet.utils.transform import random_transform_generator
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.utils import create_folder, image_path, annotation_path, root_dir, DEBUG_MODE
import keras.models
from object_detector_retinanet.utils import replace_env_vars
from keras.utils import get_file
import keras_metrics


def download_imagenet(depth):
    """ Downloads ImageNet weights and returns path to weights file.
    """
    resnet_filename = 'ResNet-{}-model.keras.h5'
    resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

    filename = resnet_filename.format(depth)
    resource = resnet_resource.format(depth)
    if depth == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif depth == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif depth == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def load_model(filepath, convert=False):
    model = keras.models.load_model(filepath)
    if convert:
        model.trainable = False

    return model


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_model(layers, num_classes, weights):
    """ Creates two models (model, training_model).

    Args
        layers             : Amount of layers
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
    """

    inputs = keras.layers.Input(shape=(None, None, 3))
    if num_classes == 1:
        num_classes += 1

    # create the resnet backbone
    if layers == 50:
        model = keras_resnet.models.ResNet50(inputs, classes=num_classes, include_top=True, freeze_bn=True)
    elif layers == 101:
        model = keras_resnet.models.ResNet101(inputs, classes=num_classes, include_top=True, freeze_bn=True)
    elif layers == 152:
        model = keras_resnet.models.ResNet152(inputs, classes=num_classes, include_top=True, freeze_bn=True)
    else:
        raise ValueError('Layers (\'{}\') is invalid.'.format(layers))

    model = model_with_weights(model, weights=weights,
                               skip_mismatch=True)

    # compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001),
        metrics=[
            keras_metrics.precision(), keras_metrics.recall()
        ]
    )

    return model


def create_callbacks(model, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    tensorboard_update_freq = 'epoch'
    if args.tensorboard_update_freq is not None:
        tensorboard_update_freq = args.tensorboard_update_freq

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            update_freq=tensorboard_update_freq
        )
        callbacks.append(tensorboard_callback)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                'resnet{layers}_{dataset_type}_{{epoch:02d}}.h5'.format(layers=args.layers,
                                                                        dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks


def create_generators(args):
    """ Create generators for training and validation.
    """
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        logging.info(
            f'Using augmentations tactic: {args.augmentations_tactic}.')
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'csv':
        train_generator = CSVClassifierGenerator(
            args.annotations,
            args.classes,
            base_dir=args.base_dir,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            images_cls_cache_path=args.images_cls_cache,
            max_annotations=args.max_annotations,
            augmentations_tactic=args.augmentations_tactic
        )

        if args.val_annotations:
            validation_generator = CSVClassifierGenerator(
                args.val_annotations,
                args.classes,
                base_dir=args.base_dir,
                batch_size=args.batch_size,
                image_min_side=args.image_min_side,
                image_max_side=args.image_max_side,
                images_cls_cache_path=args.images_cls_cache,
                max_annotations=args.max_annotations
            )
        else:
            validation_generator = None
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Simple training script for training a ResNet network.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument(
        'coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument(
        'pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument(
        'kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument(
        '--version', help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument(
        '--labels-filter', help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir',
                            help='Path to store annotation cache.', default='.')
    oid_parser.add_argument(
        '--fixed-labels', help='Use the exact specified labels.', default=False)

    data_dir = annotation_path()
    args_annotations = os.path.join(data_dir, 'annotations_train.csv')
    args_classes = os.path.join(data_dir, 'class_mappings_train.csv')
    args_val_annotations = os.path.join(data_dir, 'annotations_val.csv')

    args_snapshot_path = os.path.join(root_dir(), 'snapshot')
    args_tensorboard_dir = os.path.join(root_dir(), 'logs')
    args_images_cls_cache = os.path.join(root_dir(), 'images_cls_cache')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('--annotations', help='Path to CSV file containing annotations for training.',
                            default=args_annotations)
    csv_parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sku_class_mappings.csv'))
    csv_parser.add_argument('--val-annotations',
                            help='Path to CSV file containing annotations for validation (optional).',
                            default=args_val_annotations)
    csv_parser.add_argument('--base_dir',
                            help='Path to base dir for CSV file.',
                            default=image_path())

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',
                       help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
                       action='store_const', const=True, default=True)
    group.add_argument(
        '--weights', help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights',
                       action='store_const', const=False)

    parser.add_argument(
        '--layers', help='ResNet layers.', default=50, choices=[50, 101, 152])
    parser.add_argument(
        '--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument(
        '--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument(
        '--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.',
                        action='store_true')
    parser.add_argument(
        '--epochs', help='Number of epochs to train.', type=int, default=150)
    parser.add_argument(
        '--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
                        default=args_snapshot_path)
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default=args_tensorboard_dir)
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.',
                        dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument(
        '--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument(
        '--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--augmentations-tactic',
                        help='Tactic to which perform augmentations.',
                        default='random',
                        choices=['random'])
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=400)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=666)
    parser.add_argument('--images-cls-cache',
                        help='Path to store images classes cache (for faster loading when images are stored in the cloud)',
                        default=args_images_cls_cache)
    parser.add_argument(
        '--max-annotations', help='Trim annotations to max number (easier debugging)', type=int)
    parser.add_argument('--tensorboard-update-freq',
                        help='Number of batches frequency to update tensorboard', type=int)

    return check_args(parser.parse_args(args))


def main(args=None):
    configure_logging()
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    replace_env_vars(args)

    if DEBUG_MODE:
        args.image_min_side = 200
        args.image_max_side = 200
        args.steps = 10

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

    # Weights and logs saves in a new locations
    stmp = time.strftime("%c").replace(":", "_").replace(" ", "_")
    args.snapshot_path = os.path.join(args.snapshot_path, stmp)
    args.tensorboard_dir = os.path.join(args.tensorboard_dir, stmp)
    logging.info("Weights will be saved in  {}".format(args.snapshot_path))
    logging.info("Logs will be saved in {}".format(args.tensorboard_dir))
    create_folder(args.snapshot_path)
    create_folder(args.tensorboard_dir)

    # create the generators
    train_generator, validation_generator = create_generators(args)
    logging.info('train_size:{},val_size:{}'.format(
        train_generator.size(), validation_generator.size()))

    # create the model
    if args.snapshot is not None:
        logging.info('Loading model, this may take a second...')
        model = load_model(args.snapshot)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = download_imagenet(args.layers)

        logging.info('Creating model, this may take a second...')

        model = create_model(
            layers=args.layers,
            num_classes=train_generator.num_classes(),
            weights=weights
        )

    # print model summary
    # print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        args
    )

    # start training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=validation_generator.size()
    )


if __name__ == '__main__':
    main()
