import os
import sys
import argparse
import logging
import csv
import datetime
import pandas as pd
import numpy as np
from object_detector_retinanet.keras_retinanet.bin.predict import main as predict
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.keras_retinanet.utils.to_coco import load_annotations_to_df
from object_detector_retinanet.keras_retinanet.utils import EmMerger


class PseudoLabeling:
    def __init__(self, gt_path, dt_path, hard_score_rate):
        gt_df = load_annotations_to_df(args.annotations, 'ground-truths')
        gt_df['confidence'] = 1
        gt_df['hard_score'] = 1

        dt_df = load_annotations_to_df(args.annotations, 'detections')

        self.gt_df = gt_df
        self.dt_df = dt_df
        self.hard_score_rate = hard_score_rate
        self.out_dir = out_dir

    def em_merger(self):
        timestamp = datetime.datetime.utcnow()
        res_file = os.path.join(self.out_dir, 'detections_output_iou_{}_{}.csv'.format(
            self.hard_score_rate, timestamp))
        image_names_gt = list(set(gt_df['image_name']))
        csv_data_lst = []
        csv_data_lst.append(['image_id', 'x1', 'y1', 'x2',
                             'y2', 'confidence', 'hard_score'])
        for image_name in image_names_gt:
            height = gt_df['height']
            width = gt_df['width']

            df = pd.concat([gt_df[image_name], dt_df[image_name]])

            image_boxes = df[['x1', 'y2', 'x2', 'y2']].to_numpy()
            image_scores = df['confidence'].to_numpy()
            image_hard_scores = df['hard_score'].to_numpy()
            image_labels = np.zeros((len(df.index), 1))
            results = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
                 np.expand_dims(image_labels, axis=1)], axis=1)

            filtered_data = EmMerger.merge_detections(
                '', image_name, (height, width, 3), results, self.hard_score_rate)

            for ind, detection in filtered_data.iterrows():
                row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                       detection['confidence'], detection['hard_score']]
                csv_data_lst.append(row)

        # Save annotations csv file
        with open(res_file, 'w') as fl_csv:
            writer = csv.writer(fl_csv)
            writer.writerows(csv_data_lst)
        logging.info(f'Saved output file at: {res_file}')


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
    parser.add_argument('--out', help='Path to out dir results.')

    return parser.parse_args(args)


if __name__ == '__main__':
    configure_logging()
    args = sys.argv[1:]
    args = parse_args(args)

    pl = PseudoLabeling(
        args.annotations, args.predicted_annotations, args.hard_score_rate, args.out)
    if args.dup_removal_tactic == 'em-merger':
        pl.em_merger()
