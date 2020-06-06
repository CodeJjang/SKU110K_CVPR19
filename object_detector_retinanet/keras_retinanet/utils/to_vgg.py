import argparse
import os
import ntpath
import csv
import json
import pandas as pd
from tqdm import tqdm
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_last_folder, get_path_fname, rm_dir_content


class VGG:
    """A class responsible for converting the given CSV annotations to VGG Via tool"""

    def __init__(self, annotations_path, annotations_fname, images_path, images_folder, annotations_df, classes):
        self.annotations_path = annotations_path
        self.annotations_fname = annotations_fname
        annotations_folder = os.path.dirname(annotations_path)
        self.output_dir_path = os.path.join(
            annotations_folder, annotations_fname)
        self.output_file_path = os.path.join(
            self.output_dir_path, annotations_fname + '.csv')
        self.images_path = images_path
        self.images_folder = images_folder
        self.annotations_df = annotations_df
        self.classes = classes
        self.csv_headers = ['#filename', 'file_size', 'file_attributes',
                            'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']

    def _gen_annotations(self):
        csv_rows = []
        img_objects_count = {}
        for (_, image_fname, x1, y1, x2, y2, class_name, _, _) in tqdm(
                self.annotations_df.itertuples(), total=self.annotations_df.shape[0]):

            csv_row = []
            file_size = self._get_image_size_bytes(
                os.path.join(self.images_path, image_fname))
            file_attributes = str({})
            region_attributes = json.dumps({"object": class_name})

            if image_fname not in img_objects_count:
                img_objects_count[image_fname] = 1
            else:
                img_objects_count[image_fname] += 1

            # ids starts from 0
            region_id = img_objects_count[image_fname] - 1
            region_shape_attributes = json.dumps({
                "name": "rect",
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })

            csv_row = [image_fname, file_size, file_attributes,
                       region_id, region_shape_attributes, region_attributes]
            csv_rows.append(csv_row)

        for row in csv_rows:
            image_fname = row[0]
            row.insert(3, img_objects_count[image_fname])

        print('Saving to file...')
        with open(self.output_file_path, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(self.csv_headers)
            wr.writerows(csv_rows)
        print(
            f'Generated {len(self.annotations_df)} VGG annotations for {len(img_objects_count)} images')

    def _assert_dirs(self):
        print('Asserting and creating dirs...')
        output_dir_path = self.output_dir_path
        create_dirpath_if_not_exist(self.annotations_path)
        remove_old_output = input(
            f'Clear old output directory? [Y\\n] ({output_dir_path})')
        if remove_old_output is 'Y':
            rm_dir_content(output_dir_path)
        create_dirpath_if_not_exist(output_dir_path)

    def _get_image_size_bytes(self, path):
        return os.stat(path).st_size

    def convert(self):
        self._assert_dirs()
        print('Starting conversion...')
        self._gen_annotations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts CSV annotations to VGG format')
    parser.add_argument('--data', dest='csv_annotations_path', type=str,
                        help='csv annotations file full path')
    parser.add_argument('--images', dest='images_path', type=str,
                        help='images full path')
    args = parser.parse_args()

    columns = ['image_name', 'x1', 'y1', 'x2', 'y2',
               'class', 'image_width', 'image_height']
    annotations_df = pd.read_csv(args.csv_annotations_path, names=columns)
    classes = list(set(annotations_df['class']))

    # This name will be used to output the folder with all the outputs
    csv_annotations_name = os.path.splitext(get_path_fname(args.csv_annotations_path))
    if len(csv_annotations_name) is not 2:
        raise ValueError(
            'The passed --data argument does not lead to a valid file name')
    csv_annotations_name = f'{csv_annotations_name[0]}_vgg'

    images_path = args.images_path
    images_folder = get_last_folder(images_path)

    vgg = VGG(args.csv_annotations_path, csv_annotations_name,
              args.images_path, images_folder,
              annotations_df, classes)
    vgg.convert()
