"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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

from .generator import Generator
from object_detector_retinanet.keras_retinanet.preprocessing.get_image_size import get_image_size
from ..utils.image import read_image_bgr
import object_detector_retinanet.utils as utils

import cv2
import numpy as np
from PIL import Image
from six import raise_from
from tqdm import tqdm
import pickle

import csv
import sys
import os
import logging
from object_detector_retinanet.utils import is_path_exists, trim_csv_to_lines


class CSVClassifierGenerator(CSVGenerator):
    """ Generate data for a custom CSV dataset for classification task (image patches).
    """

    def __init__(
            self,
            **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        super(CSVClassifierGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """

        image = self.image_existence.get(self.image_path(image_index), None)
        if image is None:
            logging.error("Error: Image path {} is not existed".format(
                self.image_path(image_index)))

        # return float(2448) / float(3264)
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annots = self.image_data[path]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes
