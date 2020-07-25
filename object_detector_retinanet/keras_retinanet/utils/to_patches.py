import argparse
import os
import ntpath
import xml.etree.cElementTree as ET
import pandas as pd
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_last_folder, get_path_fname, rm_dir_content, to_csv
from object_detector_retinanet.keras_retinanet.utils.to_coco import load_annotations_to_df
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr
from tqdm import tqdm
from PIL import Image
import cv2


class ImagePatcher:
    """A class responsible for cropping the images dataset to patches"""

    def __init__(self, images_path, train_gt_path, val_gt_path, test_gt_path, images_out_dir, csv_out_dir):
        train_gt_df = load_annotations_to_df(train_gt_path, 'ground-truths')
        val_gt_df = load_annotations_to_df(val_gt_path, 'ground-truths')
        test_gt_df = load_annotations_to_df(test_gt_path, 'ground-truths')

        self.images_path = images_path
        self.train_gt_df = train_gt_df
        self.train_fname = get_path_fname(train_gt_path)
        self.val_gt_df = val_gt_df
        self.val_fname = get_path_fname(val_gt_path)
        self.test_gt_df = test_gt_df
        self.test_fname = get_path_fname(test_gt_path)
        self.images_out_dir = images_out_dir
        self.csv_out_dir = csv_out_dir

    def _gen_patches(self, gt_df, fname):
        res_file = os.path.join(self.csv_out_dir, f'patches_{fname}')
        image_names_gt = list(set(gt_df['image_name']))
        csv_rows = []
        for idx, image_name in tqdm(enumerate(image_names_gt)):
            image = read_image_bgr(os.path.join(self.images_path, image_name))
            image_rows = gt_df.loc[gt_df['image_name'] == image_name]

            height = image_rows['image_height'].values[0]
            width = image_rows['image_width'].values[0]

            for _, row in image_rows.iterrows():
                x1 = row['x1']
                y1 = row['y1']
                x2 = row['x2']
                y2 = row['y2']
                patch = image[y1: y2, x1: x2]

                img_name_start, img_name_end = image_name.split('.')
                new_img_name = f'{img_name_start}_{idx}.{img_name_end}'
                row = [new_img_name, x1, y1, x2, y2, 'object', width, height]
                csv_rows.append(row)
                cv2.imwrite(os.path.join(
                    self.images_out_dir, new_img_name), patch)

        to_csv(res_file, csv_rows)

    def _assert_dirs(self, dirs):
        print('Asserting and creating dirs...')
        output_dir_path = dirs
        remove_old_output = input(
            f'Clear old output directory? [Y\\n] ({output_dir_path})')
        if remove_old_output is 'Y':
            rm_dir_content(output_dir_path)
        create_dirpath_if_not_exist(output_dir_path)

    def extract(self):
        self._assert_dirs(self.images_out_dir)
        self._assert_dirs(self.csv_out_dir)
        print('Starting extraction...')
        self._gen_patches(self.train_gt_df, self.train_fname)
        self._gen_patches(self.val_gt_df, self.val_fname)
        # self._gen_patches(self.test_gt_df, self.test_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracts images patches from dataset and saves as csv')
    parser.add_argument(
        '--train-annotations', help='Path for training ground truth annotations CSV.')
    parser.add_argument(
        '--val-annotations', help='Path for validation ground truth annotations CSV.')
    parser.add_argument(
        '--test-annotations', help='Path for test ground truth annotations CSV.')
    parser.add_argument('--images-out', help='Path to out dir csv.')
    parser.add_argument('--csv-out', help='Path to out dir patches.')
    parser.add_argument('--images', dest='images_path', type=str,
                        help='images full path')
    args = parser.parse_args()

    images_path = args.images_path
    images_folder = get_last_folder(images_path)

    ip = ImagePatcher(images_path, args.train_annotations,
                      args.val_annotations, args.test_annotations, args.images_out, args.csv_out)
    ip.extract()
