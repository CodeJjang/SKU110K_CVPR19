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
        save_path=None,
        out_dir=None,
        flush_csv_freq=None,
        res_file=None):
    csv_data_lst = []
    result_dir = os.path.join(out_dir, 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()

    if res_file is None:
        csv_data_lst.append(['image_id', 'class', 'score'])
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

        # run network
        scores = model.predict_on_batch(
            np.expand_dims(image, axis=0))

        scores = scores.squeeze()
        # find the order with which to sort the scores
        pred_idx = np.argmax(scores)
        label = generator.label_to_name(pred_idx)

        score = scores[pred_idx]

        row = [image_name, label, score]
        csv_data_lst.append(row)

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
