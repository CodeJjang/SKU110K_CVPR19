import argparse
import os
import ntpath
import xml.etree.cElementTree as ET
import pandas as pd


class PascalVoc:
    """A class responsible for converting the given CSV annotations to Pascal VOC XMLs for displaying in LabelImg tool"""

    def __init__(self, annotations_path, annotations_fname, images_path, images_folder, annotations_df, classes):
        self.annotations_path = annotations_path
        self.annotations_fname = annotations_fname
        annotations_folder = os.path.dirname(annotations_path)
        self.output_dir_path = os.path.join(
            annotations_folder, annotations_fname)
        self.images_path = images_path
        self.images_folder = images_folder
        self.annotations_df = annotations_df
        self.classes = classes
        self.images_channels = 3

    def _gen_annotations(self):
        images_dict = {}
        for idx, row in self.annotations_df.iterrows():
            image_fname = row['image_name']
            image_name = image_fname.split('.')[0]

            # In case this is the first time we tackle this image, create it's XML root
            if image_name not in images_dict:
                root = ET.Element("annotation")
                ET.SubElement(root, "folder").text = self.images_folder
                ET.SubElement(root, "filename").text = image_fname
                ET.SubElement(root, "path").text = os.path.join(
                    self.images_path, image_fname)

                source = ET.SubElement(root, "source")
                ET.SubElement(source, "database").text = "SKU110K"

                size = ET.SubElement(root, "size")
                ET.SubElement(size, "width").text = str(row['image_width'])
                ET.SubElement(size, "height").text = str(row['image_height'])
                ET.SubElement(size, "depth").text = str(self.images_channels)

                ET.SubElement(root, "segmented").text = str(0)

                images_dict[image_name] = root
            else:
                root = images_dict[image_name]

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = row['class']
            ET.SubElement(obj, "pose").text = 'Unspecified'
            ET.SubElement(obj, "truncated").text = str(0)
            ET.SubElement(obj, "difficult").text = str(0)
            bb = ET.SubElement(obj, "bndbox")
            xmin, ymin, xmax, ymax = row['x1'], row['y1'], row['x2'], row['y2']
            ET.SubElement(bb, 'xmin').text, \
                ET.SubElement(bb, 'ymin').text, \
                ET.SubElement(bb, 'xmax').text, \
                ET.SubElement(bb, 'ymax').text = str(
                    xmin), str(ymin), str(xmax), str(ymax)

        for image_name, root in images_dict.items():
            tree = ET.ElementTree(root)
            fpath = os.path.join(self.output_dir_path, image_name + '.xml')
            tree.write(fpath)
        print(
            f'Generated {len(self.annotations_df)} Pascal annotations for {len(images_dict)} images')

    def _create_dirpath_if_not_exist(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _rm_dir(self, dir_path):
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

    def _assert_dirs(self):
        print('Asserting and creating dirs...')
        output_dir_path = self.output_dir_path
        self._create_dirpath_if_not_exist(self.annotations_path)
        remove_old_output = input(
            f'Clear old output directory? [Y\\n] ({output_dir_path})')
        if remove_old_output is 'Y':
            self._rm_dir(output_dir_path)
        self._create_dirpath_if_not_exist(output_dir_path)

    def convert(self):
        self._assert_dirs()
        print('Starting conversion...')
        self._gen_annotations()


def get_path_fname(path):
    '''
    Extract basename from file path
    '''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_last_folder(path):
    return os.path.basename(os.path.normpath(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts CSV annotations to Pascal VOC format')
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
    csv_annotations_name = get_path_fname(args.csv_annotations_path).split('.')
    if len(csv_annotations_name) is not 2:
        raise ValueError(
            'The passed --data argument does not lead to a valid file name')
    csv_annotations_name = csv_annotations_name[0]

    images_path = args.images_path
    images_folder = get_last_folder(images_path)

    pascal_voc = PascalVoc(args.csv_annotations_path, csv_annotations_name,
                           args.images_path, images_folder,
                           annotations_df, classes)
    pascal_voc.convert()
