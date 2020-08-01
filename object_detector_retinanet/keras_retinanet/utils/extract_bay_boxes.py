import os
import sys
import argparse
import logging
import csv
import datetime
import pandas as pd
import numpy as np
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.keras_retinanet.utils.to_coco import load_annotations_to_df
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_path_base_path, get_path_fname, to_csv


class BayBoxExtractor:
    def __init__(self, train_gt_path, val_gt_path, test_gt_path, out_dir):
        train_gt_df = load_annotations_to_df(train_gt_path, 'ground-truths')
        val_gt_df = load_annotations_to_df(val_gt_path, 'ground-truths')
        test_gt_df = load_annotations_to_df(test_gt_path, 'ground-truths')

        self.train_gt_df = train_gt_df
        self.train_fname = get_path_fname(train_gt_path)
        self.val_gt_df = val_gt_df
        self.val_fname = get_path_fname(val_gt_path)
        self.test_gt_df = test_gt_df
        self.test_fname = get_path_fname(test_gt_path)
        self.out_dir = out_dir

    def _extract_annotations(self, df, fname):
        res_file = os.path.join(self.out_dir, f'bay_{fname}')
        image_names_gt = list(set(df['image_name']))
        csv_rows = []
        for image_name in image_names_gt:
            height = df.loc[df['image_name'] ==
                            image_name]['image_height'].values[0]
            width = df.loc[df['image_name'] ==
                           image_name]['image_width'].values[0]

            x1 = min(df.loc[df['image_name'] == image_name]['x1'].to_numpy())
            y1 = min(df.loc[df['image_name'] == image_name]['y1'].to_numpy())
            x2 = max(df.loc[df['image_name'] == image_name]['x2'].to_numpy())
            y2 = max(df.loc[df['image_name'] == image_name]['y2'].to_numpy())
            row = [image_name, x1, y1, x2, y2, 'object', width, height]
            csv_rows.append(row)

        to_csv(res_file, csv_rows)
        

    def extract(self):
        create_dirpath_if_not_exist(self.out_dir)
        self._extract_annotations(self.train_gt_df, self.train_fname)
        self._extract_annotations(self.val_gt_df, self.val_fname)
        self._extract_annotations(self.test_gt_df, self.test_fname)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Extract bay boxes from SKU box annotations.')

    parser.add_argument(
        '--annotations', help='Path for training ground truth annotations CSV.')
    parser.add_argument(
        '--val-annotations', help='Path for validation ground truth annotations CSV.')
    parser.add_argument(
        '--test-annotations', help='Path for test ground truth annotations CSV.')
    parser.add_argument('--out', help='Path to out dir results.')

    return parser.parse_args(args)


if __name__ == '__main__':
    configure_logging()
    args = sys.argv[1:]
    args = parse_args(args)

    bbe = BayBoxExtractor(
        args.annotations, args.val_annotations, args.test_annotations, args.out)
    bbe.extract()
