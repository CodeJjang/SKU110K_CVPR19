import argparse
import os
import ntpath
import json
import pandas as pd
import numpy as np
from pycocotools.cocoeval import COCOeval
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_last_folder, get_path_fname, rm_dir


class JsonCOCO:
    """Helper class to hold COCO json in memory instead of file"""

    def __init__(self, list_of_coco_dicts):
        self.list_of_coco_dicts = list_of_coco_dicts

    def getImgIds(self):
        return list(set([e['image_id'] for e in self.list_of_coco_dicts]))

    def getCatIds(self):
        return list(set([e['category_id'] for e in self.list_of_coco_dicts]))

    def getAnnIds(self, imgIds, catIds):
        AnnIds = [element['id'] for element in self.list_of_coco_dicts if element['image_id']
                  in imgIds and element['category_id'] in catIds]
        return AnnIds

    def loadAnns(self, AnnIds):
        list_of_filtered_dicts = [
            element for element in self.list_of_coco_dicts if element['id'] in AnnIds]
        return list_of_filtered_dicts


class COCO:
    """A class responsible for converting the given CSV annotations to COCO jsons for calculating COCO evaluations"""

    def __init__(self, annotations_path, annotations_fname, annotations_df, needs_score, should_output_file=False):
        self.annotations_path = annotations_path
        self.annotations_fname = annotations_fname
        annotations_folder = os.path.dirname(annotations_path)
        self.output_dir_path = os.path.join(
            annotations_folder, annotations_fname)
        self.annotations_df = annotations_df
        self.needs_score = needs_score
        self.should_output_file = should_output_file

    def _get_base_coco_dict(self):
        return {
            "segmentation": None,
            "iscrowd": 0,
            "image_id": None,
            "category_id": 1,
            "id": None,
            "bbox": None,
            "area": None
        }

    def _gen_annotations(self):
        out_annotations = []
        img_list = list(set(self.annotations_df['image_name'].tolist()))

        for idx, row in self.annotations_df.iterrows():
            entry = self._get_base_coco_dict()
            entry['image_id'] = img_list.index(row['image_name'])
            entry['id'] = idx
            xmin, ymin, xmax, ymax = row['x1'], row['y1'], row['x2'], row['y2']
            width = xmax - xmin
            height = ymax - ymin
            entry['bbox'] = [xmin, ymin, width, height]
            entry['area'] = width * height

            if self.needs_score:
                entry['score'] = row['hard_score']

            out_annotations.append(entry)

        if self.should_output_file:
            fpath = os.path.join(self.output_dir_path,
                                 self.annotations_fname + '.json')
            with open(fpath, 'w') as f:
                json.dump(out_annotations, f)
        print(
            f'Generated {len(self.annotations_df)} COCO annotations for {len(img_list)} images')
        return out_annotations

    def _assert_dirs(self):
        print('Asserting and creating dirs...')
        output_dir_path = self.output_dir_path
        create_dirpath_if_not_exist(self.annotations_path)
        remove_old_output = input(
            f'Clear old output directory? [Y\\n] ({output_dir_path})')
        if remove_old_output is 'Y':
            rm_dir(output_dir_path)
        create_dirpath_if_not_exist(output_dir_path)

    def convert(self):
        if self.should_output_file:
            self._assert_dirs()
        print('Starting conversion...')
        return self._gen_annotations()


def get_annotations_columns(data_type):
    columns = ['image_name', 'x1', 'y1', 'x2', 'y2']
    if data_type == 'ground-truths':
        columns += ['class', 'image_width', 'image_height']
    elif data_type == 'detections':
        columns += ['confidence', 'hard_score']
    else:
        raise ValueError(
            f'Unsupported {data_type} passed to get_annotations_columns')
    return columns


def load_annotations_to_df(csv_annotations_path, data_type, max_gt_annotations=None):
    init_row = 0
    # Skip first row of detections as it's column names
    if data_type == 'detections':
        init_row = 1
    columns = get_annotations_columns(data_type)
    return pd.read_csv(csv_annotations_path, skiprows=init_row, names=columns, nrows=max_gt_annotations)


def summarize(eval_results, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = eval_results.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = eval_results.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = eval_results.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


def print_metrics(gt_annotations_path, dt_annotations_path, max_gt_annotations):

    # Get GT COCO json
    gt_annotations_df = load_annotations_to_df(
        gt_annotations_path, 'ground-truths', max_gt_annotations)
    gt_json = COCO(gt_annotations_path, '',
                   gt_annotations_df, needs_score=False, should_output_file=False).convert()

    # Get detections COCO json
    dt_annotations_df = load_annotations_to_df(
        dt_annotations_path, 'detections')
    dt_json = COCO(dt_annotations_path, '',
                   dt_annotations_df, needs_score=True, should_output_file=False).convert()

    gt_coco_format = JsonCOCO(gt_json)
    dt_coco_format = JsonCOCO(dt_json)

    # running evaluation
    cocoEval = COCOeval(gt_coco_format, dt_coco_format, iouType='bbox')
    cocoEval.params.areaRng = [[0, 100000000]]
    cocoEval.params.areaRngLbl = ['all']
    cocoEval.params.maxDets = [300]
    cocoEval.evaluate()
    cocoEval.accumulate()
    summarize(cocoEval, ap=1, maxDets=300)
    summarize(cocoEval, ap=1, maxDets=300, iouThr=0.5)
    summarize(cocoEval, ap=1, maxDets=300, iouThr=0.75)
    summarize(cocoEval, ap=0, maxDets=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts CSV annotations to COCO format')
    parser.add_argument('--data', dest='csv_annotations_path', type=str,
                        help='csv annotations file full path')
    parser.add_argument('--type', dest='data_type', choices=['detections', 'ground-truths'],
                        help='whether it\'s detections or ground truths')
    args = parser.parse_args()

    annotations_df = load_annotations_to_df(
        args.csv_annotations_path, args.data_type)

    # This name will be used to output the folder with all the outputs
    csv_annotations_name = os.path.splitext(
        get_path_fname(args.csv_annotations_path))
    if len(csv_annotations_name) is not 2:
        raise ValueError(
            'The passed --data argument does not lead to a valid file name')
    csv_annotations_name = csv_annotations_name[0]

    # Whether to include score to output or not
    needs_score = args.data_type == 'detections'
    should_output_file = True
    coco = COCO(args.csv_annotations_path, csv_annotations_name,
                annotations_df, needs_score, should_output_file)
    coco.convert()
