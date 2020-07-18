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

import csv

import datetime
import logging
import tensorflow as tf
from object_detector_retinanet.utils import append_csv, create_folder, is_path_exists, load_csv, write_csv
from .visualization import draw_detections, draw_annotations
import numpy as np
import os
from tqdm import tqdm
import cv2

predictions_cache = {}


def load_saved_image_names(csv_path):
    if not is_path_exists(csv_path):
        return []
    lines = load_csv(csv_path)
    # Remove header columns
    lines = lines[1:]
    return list(set([line[0] for line in lines]))


def predict(
        generator,
        model,
        score_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections=9999,
        save_path=None,
        base_dir=None,
        out_dir=None,
        predict_from_cache=None,
        flush_csv_freq=None,
        res_file=None):
    csv_data_lst = []
    result_dir = os.path.join(out_dir, 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()

    if res_file is None:
        csv_data_lst.append(['image_id', 'x1', 'y1', 'x2',
                             'y2', 'hard_score'])
        res_file = os.path.join(
            result_dir, f'detections_output_{timestamp}.csv')
        logging.info(f'Output file: {res_file}')

    if flush_csv_freq is not None:
        image_names = load_saved_image_names(res_file)

    if save_path is not None:
        save_path += str(timestamp)

    for i in tqdm(range(generator.size())):
        image_name = generator.image_path(i).split(os.path.sep)[-1]

        # Skip image if we're flushing images to csv and we've seen this image
        if flush_csv_freq is not None and image_name in image_names:
            continue

        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # We use cached predictions when we run 'predict' many times from a script, to calculate metrics
        if predict_from_cache and image_name in predictions_cache:
            boxes, scores, labels = predictions_cache[image_name]
            boxes, scores, labels = boxes.copy(), scores.copy(), labels.copy()
        else:
            # run network
            boxes, scores, labels = model.predict_on_batch(
                np.expand_dims(image, axis=0))
            if predict_from_cache:
                predictions_cache[image_name] = boxes.copy(
                ), scores.copy(), labels.copy()

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        scores = scores[scores_sort]
        labels = labels[0, indices[scores_sort]]

        indices = tf.image.non_max_suppression(
            image_boxes, scores, max_detections, iou_threshold=nms_iou_threshold,
            score_threshold=0.1
        )
        indices = tf.Session().run(indices)
        image_boxes = image_boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for det_idx, detection in enumerate(image_boxes):
            box = np.asarray([detection[0], detection[1],
                              detection[2], detection[3]])
            score = scores[det_idx]
            label = labels[det_idx]

            filtered_boxes.append(box)
            filtered_scores.append('{0:.2f}'.format(score))
            filtered_labels.append(label)

            row = [image_name, detection[0], detection[1],
                   detection[2], detection[3], score]
            csv_data_lst.append(row)

        if save_path is not None:
            create_folder(save_path)

            draw_annotations(raw_image, generator.load_annotations(
                i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
                            np.asarray(filtered_labels), color=(0, 0, 255))

            cv2.imwrite(os.path.join(save_path, image_name), raw_image)

        if flush_csv_freq is not None and (i + 1) % flush_csv_freq == 0:
            append_csv(res_file, csv_data_lst)
            csv_data_lst = []

    # Save annotations csv file
    if flush_csv_freq is None:
        write_csv(res_file, csv_data_lst)
        logging.info(f'Saved output file at: {res_file}')
    else:
        append_csv(res_file, csv_data_lst)
    return res_file
