import os
import sys
import argparse
import logging
import csv
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from object_detector_retinanet.keras_retinanet.bin.predict import main as predict
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.keras_retinanet.utils.to_coco import load_annotations_to_df
from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_path_base_path


class PseudoLabeling:
    def __init__(self, gt_path, dt_path, hard_score_rate, nms_iou_threshold, out_dir):
        gt_df = load_annotations_to_df(gt_path, 'ground-truths')
        gt_df['confidence'] = 1.
        gt_df['hard_score'] = 1.

        dt_df = load_annotations_to_df(dt_path, 'detections')

        self.gt_df = gt_df
        self.dt_df = dt_df
        self.hard_score_rate = hard_score_rate
        self.out_dir = out_dir
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = 9999

    def em_merger(self):
        timestamp = datetime.datetime.utcnow()
        res_file = os.path.join(self.out_dir, 'detections_output_iou_{}_{}.csv'.format(
            self.hard_score_rate, timestamp))
        image_names_gt = list(set(self.gt_df['image_name']))
        csv_rows = []
        for image_name in image_names_gt:
            height = self.gt_df.loc[self.gt_df['image_name']
                                    == image_name]['image_height'].values[0]
            width = self.gt_df.loc[self.gt_df['image_name']
                                   == image_name]['image_width'].values[0]

            df = pd.concat([self.gt_df.loc[self.gt_df['image_name'] == image_name],
                            self.dt_df.loc[self.dt_df['image_name'] == image_name]])

            image_boxes = df[['x1', 'y1', 'x2', 'y2']
                             ].to_numpy(dtype=np.float32)
            image_scores = df['confidence'].to_numpy(dtype=np.float32)
            image_hard_scores = df['hard_score'].to_numpy(dtype=np.float32)
            image_labels = np.zeros((len(df.index), 1), dtype=np.int32)
            results = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
                 image_labels], axis=1)

            filtered_data = EmMerger.merge_detections(
                '', image_name, (height, width, 3), results, self.hard_score_rate)

            for ind, detection in filtered_data.iterrows():
                row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                       'object', width, height]
                csv_rows.append(row)

        self.to_csv(res_file, csv_rows)

    def nms(self):
        timestamp = datetime.datetime.utcnow()
        res_file = os.path.join(self.out_dir, 'detections_output_iou_{}_{}.csv'.format(
            self.hard_score_rate, timestamp))
        image_names_gt = list(set(self.gt_df['image_name']))
        csv_rows = []
        for image_name in image_names_gt:
            height = self.gt_df.loc[self.gt_df['image_name']
                                    == image_name]['image_height'].values[0]
            width = self.gt_df.loc[self.gt_df['image_name']
                                   == image_name]['image_width'].values[0]

            df = pd.concat([self.gt_df.loc[self.gt_df['image_name'] == image_name],
                            self.dt_df.loc[self.dt_df['image_name'] == image_name]])

            image_boxes = df[['x1', 'y1', 'x2', 'y2']].to_numpy()
            image_scores = df['confidence'].to_numpy()
            image_hard_scores = df['hard_score'].to_numpy()
            scores = self.hard_score_rate * image_hard_scores + \
                (1 - self.hard_score_rate) * image_scores

            indices = tf.image.non_max_suppression(
                image_boxes, scores, self.max_detections, iou_threshold=self.nms_iou_threshold,
                score_threshold=0.1
            )
            indices = tf.Session().run(indices)
            detections = image_boxes[indices]

            for detection in detections:
                row = [image_name, detection[0], detection[1], detection[2], detection[3],
                       'object', width, height]
                csv_rows.append(row)

        self.to_csv(res_file, csv_rows)

    def to_csv(self, fname, rows):
        create_dirpath_if_not_exist(get_path_base_path(fname))

        # Save annotations csv file
        with open(fname, 'w') as fl_csv:
            writer = csv.writer(fl_csv)
            writer.writerows(rows)
        logging.info(f'Saved output file at: {fname}')


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Pseudo labeling script mixing given annotations with predicted ones.')

    parser.add_argument(
        '--annotations', help='Path for ground truth annotations CSV.')
    parser.add_argument('--predicted-annotations',
                        help='Path for predicted annotations CSV.')
    parser.add_argument('--dup-removal-tactic', help='Tactic for duplicated boxes removal.',
                        choices=['nms', 'em-merger'], default='nms')
    parser.add_argument('--hard-score-rate', help='Weight for balancing hard score and IoU score.',
                        default=0.5, type=float)
    parser.add_argument('--nms-iou-threshold', help='IoU threshold for nms.',
                        default=0.5, type=float)
    parser.add_argument('--out', help='Path to out dir results.')

    return parser.parse_args(args)


if __name__ == '__main__':
    configure_logging()
    args = sys.argv[1:]
    args = parse_args(args)

    pl = PseudoLabeling(
        args.annotations, args.predicted_annotations, args.hard_score_rate, args.nms_iou_threshold, args.out)
    if args.dup_removal_tactic == 'em-merger':
        pl.em_merger()
    elif args.dup_removal_tactic == 'nms':
        pl.nms()
