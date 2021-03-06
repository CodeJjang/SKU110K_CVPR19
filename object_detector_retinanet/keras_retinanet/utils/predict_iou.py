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
from object_detector_retinanet.keras_retinanet.utils import EmMerger
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
        max_detections=9999,
        save_path=None,
        hard_score_rate=1.,
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
                         'y2', 'confidence', 'hard_score'])
        res_file = result_dir + \
            '/detections_output_iou_{}_{}.csv'.format(
                hard_score_rate, timestamp)
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
            boxes, hard_scores, labels, soft_scores = predictions_cache[image_name]
            boxes, hard_scores, labels, soft_scores = boxes.copy(
            ), hard_scores.copy(), labels.copy(), soft_scores.copy()
        else:
            # run network
            boxes, hard_scores, labels, soft_scores = model.predict_on_batch(
                np.expand_dims(image, axis=0))
            if predict_from_cache:
                predictions_cache[image_name] = boxes.copy(
                ), hard_scores.copy(), labels.copy(), soft_scores.copy()

        soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = hard_score_rate * hard_scores + \
            (1 - hard_score_rate) * soft_scores
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(hard_scores[0, :] > score_threshold)[0]

        # select those scores
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_hard_scores = hard_scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        results = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
             np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(
            base_dir, image_name, raw_image.shape, results, hard_score_rate)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for ind, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'],
                              detection['x2'], detection['y2']])
            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                   detection['confidence'], detection['hard_score']]
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
