import argparse
import os
import ntpath
import pandas as pd
import numpy as np
from collections import Counter
from object_detector_retinanet.utils import create_dirpath_if_not_exist, get_last_folder, get_path_fname, rm_dir
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_boxes
from object_detector_retinanet.keras_retinanet.utils.image import resize_image
from bokeh.models import ColumnDataSource, Column
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral4 as palette
from bokeh.layouts import column, gridplot, row
from bokeh.models import Select
import matplotlib.patches as patches
from IPython.display import display
import random
import itertools
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setting to better display a DF in colab
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class StatisticsGenerator:
    """A class responsible for generating statistics about our SKU110K dataset"""

    def __init__(self, train_df, val_df, test_df, base_dir=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.total_df = pd.concat([train_df, val_df, test_df])
        self.all_dfs = [self.total_df,
                        self.train_df, self.val_df, self.test_df]
        self.splits = ['All', 'Train', 'Val', 'Test']
        self.base_dir = base_dir

    def vis_table(self, data_df):
        display(data_df)

    def calc_images_and_objects_amounts(self):
        columns = ['#imgs', '#objs', 'avg #objs/img']
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

        data_dfs = zip(data_dfs, self.splits, palette)
        for idx, (df, name, color) in enumerate(data_dfs):
            x_col, y_col = df.keys()
            circle = fig.circle(df[x_col], df[y_col], legend_label=name,
                                size=10, color=color, alpha=0.3)
            if idx != 0:
                circle.visible = False

        fig.legend.location = "top_right"
        fig.legend.click_policy = "hide"
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

    def get_images_with_gt_boxes(self, amount):
        df = pd.concat(self.all_dfs)
        image_names = list(set(df['image_name']))
        # Amount cannot be higher than amount of images
        amount = amount if amount <= len(image_names) else len(image_names)
        image_names = random.sample(image_names, amount)
        imgs = []
        for img_name in image_names:
            # Choose all annotations of that image
            img_rows = df.loc[df['image_name'] == img_name]
            # Get boxes
            boxes = np.array([[x1, y1, x2, y2] for _, (x1, y1, x2, y2) in img_rows[[
                             'x1', 'y1', 'x2', 'y2']].iterrows()])

            img_path = os.path.join(self.base_dir, img_name)
            img = np.asarray(Image.open(img_path).convert('RGBA'))

            # Rescale to a larger size than plot, to be able to zoom smoothly
            img, image_scale = resize_image(img, 1200, 1200)
            boxes = boxes.astype(float) * image_scale
            draw_boxes(img, boxes, (255, 0, 0, 255), thickness=2)
            imgs.append(img)

        return imgs

    def vis_images(self, imgs):
        if len(imgs) == 0:
            return

        figs = []
        for img in imgs:
            # Flip top down to be displayed correctly in bokeh
            img = img[::-1]
            width, height, _ = img.shape

            p = figure(x_range=(0, width), y_range=(0, height))

            # Hide all axes
            p.xaxis.visible = None
            p.yaxis.visible = None
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None
            p.outline_line_alpha = 0

            p.image_rgba([img], x=0, y=0, dw=width, dh=height)
            figs.append(p)

        grid = gridplot(figs, ncols=3, plot_width=400, plot_height=400)
        show(grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Outputs statistics about our dataset')
    parser.add_argument('--base_dir', type=str, required=False,
                        help='base directory for images')
    parser.add_argument('--images', type=int, required=False, default=6,
                        help='amount of images to display in --statistic plot-images')
    parser.add_argument('--train-annotations', type=str,
                        help='csv train annotations file full path')
    parser.add_argument('--val-annotations', type=str,
                        help='csv val annotations file full path')
    parser.add_argument('--test-annotations', type=str,
                        help='csv test annotations file full path')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='whether we are running in colab')
    parser.add_argument('--statistic', choices=['imgs-objs-table', 'box-areas-scatter', 'plot-images'],
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

    gen = StatisticsGenerator(train_df, val_df, test_df, args.base_dir)
    if args.statistic == 'imgs-objs-table':
        data = gen.calc_images_and_objects_amounts()
        gen.vis_table(data)
    elif args.statistic == 'box-areas-scatter':
        data = gen.calc_box_areas()
        gen.vis_scatter(data)
    elif args.statistic == 'plot-images':
        if args.base_dir is None:
            parser.error('plot-images requires --base_dir')
        data = gen.get_images_with_gt_boxes(args.images)
        gen.vis_images(data)
