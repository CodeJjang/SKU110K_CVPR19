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
from object_detector_retinanet.keras_retinanet.utils import EmMerger
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


def crop_image(image, box):
    box = np.array(box).astype(int)
    box = box[0]
    return image[box[1]: box[3], box[0]: box[2]]

def intersected(boxes, box):
    indices = []
    box = box.squeeze()
    x1, y1, x2, y2 = box
    for idx, _box in enumerate(boxes.squeeze()):
        _x1, _y1, _x2, _y2 = _box
        dx = min(x2, _x2) - max(x1, _x1)
        dy = min(y2, _y2) - max(y1, _y1)
        if (dx>=0) and (dy>=0):
            indices.append(idx)
    return np.array(indices)

def translate_boxes_origin(boxes, x_offset, y_offset):
    boxes[:, :, 0] += x_offset
    boxes[:, :, 1] += y_offset
    boxes[:, :, 2] += x_offset
    boxes[:, :, 3] += y_offset

def detect_bay(model, image, score_threshold, max_detections):
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    sorted_scores_indices = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes = boxes[0, indices[sorted_scores_indices], :]
    image_scores = scores[sorted_scores_indices]
    image_labels = labels[0, indices[sorted_scores_indices]]
    return image_boxes, image_scores, image_labels


def predict(
        objects_generator,
        bays_generator,
        bay_detection_model,
        object_detection_model,
        score_threshold=0.05,
        max_detections=9999,
        save_path=None,
        hard_score_rate=1.,
        base_dir=None,
        out_dir=None,
        flush_csv_freq=None,
        res_file=None,
        max_annotations=None):
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

    seen_annotations = 0

    for objects_gen_idx in tqdm(range(objects_generator.size())):
        image_name = objects_generator.image_path(
            objects_gen_idx).split(os.path.sep)[-1]

        # Skip image if we're flushing images to csv and we've seen this image
        if flush_csv_freq is not None and image_name in image_names:
            continue

        raw_image = objects_generator.load_image(objects_gen_idx)
        image = objects_generator.preprocess_image(raw_image.copy())
        image, scale_for_bay = objects_generator.resize_image(image)

        # run bay detector
        bay_box, bay_score, bay_label = detect_bay(
            bay_detection_model, image, score_threshold, max_detections=1)
        bay_image = crop_image(image, bay_box)
        
        # bay_image, scale_for_object_det = objects_generator.resize_image(
        #     bay_image)
        
        # run object detector
        boxes, hard_scores, labels, soft_scores = object_detection_model.predict_on_batch(
            np.expand_dims(image, axis=0))
        # boxes, hard_scores, labels, soft_scores = object_detection_model.predict_on_batch(
        #     np.expand_dims(bay_image, axis=0))

        # soft_scores = np.squeeze(soft_scores, axis=-1)
        # soft_scores = hard_score_rate * hard_scores + \
        #     (1 - hard_score_rate) * soft_scores

        # correct boxes for image scale
        boxes /= scale_for_bay
        bay_box /= scale_for_bay
        # boxes /= scale_for_object_det * scale_for_bay
        # bay_box /= scale_for_bay
        
        # get boxes indices that intersect with bay box
        boxes_indices = intersected(boxes, bay_box)
        boxes = boxes[:, boxes_indices]
        hard_scores = hard_scores[:, boxes_indices]
        labels = labels[:, boxes_indices]
        soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = soft_scores[:, boxes_indices]

        # shift boxes from bay_box coordinates to image coordinates
        # translate_boxes_origin(boxes, bay_box[0][0], bay_box[0][1])

        
        soft_scores = hard_score_rate * hard_scores + \
            (1 - hard_score_rate) * soft_scores

        # select indices which have a score above the threshold
        indices = np.where(hard_scores[0, :] > score_threshold)[0]

        # select those scores
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]

        # find the order with which to sort the scores
        sorted_scores_indices = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[sorted_scores_indices], :]
        image_scores = scores[sorted_scores_indices]
        image_hard_scores = hard_scores[sorted_scores_indices]
        image_labels = labels[0, indices[sorted_scores_indices]]
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

            # Draw bay annotations
            image_full_path = objects_generator.image_path(objects_gen_idx)
            bay_annotations = bays_generator.image_data[image_full_path][0]
            bay_annotations = np.asarray([bay_annotations['x1'], bay_annotations['y1'], bay_annotations['x2'],
                                          bay_annotations['y2'], bays_generator.classes[bay_annotations['class']]])
            draw_annotations(raw_image, np.asarray([bay_annotations]),
                             label_to_name=bays_generator.label_to_name)
            draw_detections(raw_image, bay_box, bay_score,
                            bay_label, color=(0, 0, 255), label_to_name=bays_generator.label_to_name)

            # Draw object annotations
            gt_object_annotations = objects_generator.load_annotations(objects_gen_idx)
            seen_annotations += gt_object_annotations.shape[0]
            draw_annotations(raw_image, gt_object_annotations, label_to_name=objects_generator.label_to_name)
            draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
                            np.asarray(filtered_labels), color=(0, 0, 255))

            cv2.imwrite(os.path.join(save_path, image_name), raw_image)
            
        if flush_csv_freq is not None and (objects_gen_idx + 1) % flush_csv_freq == 0:
            append_csv(res_file, csv_data_lst)
            csv_data_lst = []

        if max_annotations is not None and seen_annotations >= max_annotations:
            break

    # Save annotations csv file
    if flush_csv_freq is None:
        write_csv(res_file, csv_data_lst)
        logging.info(f'Saved output file at: {res_file}')
    else:
        append_csv(res_file, csv_data_lst)
    return res_file
