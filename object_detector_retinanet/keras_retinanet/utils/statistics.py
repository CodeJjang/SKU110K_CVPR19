import argparse
import os
import ntpath
import xml.etree.cElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_last_folder, get_path_fname, rm_dir
from bokeh.models import ColumnDataSource, Column
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
from IPython.display import display
import itertools


class StatisticsGenerator:
    """A class responsible for generating statistics about our SKU110K dataset"""

    def __init__(self, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.total_df = pd.concat([train_df, val_df, test_df])
        self.all_dfs = [self.train_df, self.val_df,
                        self.test_df, self.total_df]
        self.splits = ['Train', 'Val', 'Test', 'All']

    def vis_table(self, data_df):
        display(data_df)

    def calc_images_and_objects_amounts(self):
        columns = ['#imgs', '#objects', 'avg #objects/imgs']
        columns = pd.MultiIndex.from_tuples(
            [(split, col) for split in self.splits for col in columns])

        data = []
        for df in self.all_dfs:
            imgs_amount = len(set(df['image_name']))
            objects_amount = len(df)
            objects_per_image = Counter(df['image_name'])
            avg_objects_per_image = int(
                np.floor(np.mean(list(Counter(objects_per_image).values()))))
            data += [imgs_amount, objects_amount, avg_objects_per_image]

        return pd.DataFrame(data=np.array([data]), columns=columns)

    def vis_scatter(self, data_dfs):
        if len(data_dfs) < 1:
            return

        fig = figure()
        fig.xaxis.axis_label = data_dfs[0].keys()[0]
        fig.yaxis.axis_label = data_dfs[0].keys()[1]

        data_dfs = zip(data_dfs, itertools.cycle(palette))
        for df, color in data_dfs:
            x_col, y_col = df.keys()
            fig.circle(x=x_col, y=y_col,
                       source=ColumnDataSource(df),
                       size=10, color=color, alpha=0.5)
            break
        show(fig)

    def calc_box_areas(self):
        columns = ['Box areas (sqrt)', 'Amount']
        data = []
        for df in self.all_dfs:
            areas = (df['x2'].to_numpy() - df['x1'].to_numpy()) * \
                (df['y2'].to_numpy() - df['y1'].to_numpy())
            areas_count = Counter(areas)
            areas = areas_count.keys()
            areas_amount = areas_count.values()
            x = np.sqrt(np.array(list(areas))).round()
            y = np.array(list(areas_amount))
            sorted_indices = x.argsort()
            x = x[sorted_indices]
            y = y[sorted_indices]
            data.append(pd.DataFrame(data=np.array([x, y]).T, columns=columns))

        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Outputs statistics about our dataset')
    parser.add_argument('--train-annotations', type=str,
                        help='csv train annotations file full path')
    parser.add_argument('--val-annotations', type=str,
                        help='csv val annotations file full path')
    parser.add_argument('--test-annotations', type=str,
                        help='csv test annotations file full path')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='whether we are running in colab')
    parser.add_argument('--statistic', choices=['imgs-objs-table', 'box-areas-scatter'],
                        help='statistic to print')
    args = parser.parse_args()

    if args.colab:
        # Loads BokehJS
        output_notebook()

    columns = ['image_name', 'x1', 'y1', 'x2', 'y2',
               'class', 'image_width', 'image_height']
    train_df = pd.read_csv(args.train_annotations, names=columns)
    val_df = pd.read_csv(args.val_annotations, names=columns)
    test_df = pd.read_csv(args.test_annotations, names=columns)

    gen = StatisticsGenerator(train_df, val_df, test_df)
    if args.statistic == 'imgs-objs-table':
        data = gen.calc_images_and_objects_amounts()
        gen.vis_table(data)
    elif args.statistic == 'box-areas-scatter':
        data = gen.calc_box_areas()
        gen.vis_scatter(data)
