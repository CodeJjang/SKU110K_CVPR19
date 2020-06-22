import os
import sys
import argparse
import logging
import numpy as np

from object_detector_retinanet.keras_retinanet.bin.predict import main as predict
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.utils import create_dirpath_if_not_exist, rm_dir, assign_to_args
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from IPython.display import clear_output


def vis_plot(hard_score_rates, average_precisions):
    fig = figure()
    fig.xaxis.axis_label = 'hard_score_rate'
    fig.yaxis.axis_label = 'AP@[ IoU=0.50:0.95 | area= all | maxDets=300 ]'

    fig.line(hard_score_rates, average_precisions, line_width=2)
    show(fig)


if __name__ == '__main__':
    configure_logging()
    args = sys.argv[1:]

    if '--colab' in args:
        # Loads BokehJS
        output_notebook()
        # Remove, because later 'predict()' doesn't expect that value
        args.remove('--colab')

    # Inject temp_folder to all temp created files, so we can delete it in the end
    temp_folder_path = './temp'
    create_dirpath_if_not_exist(temp_folder_path)
    assign_to_args(args, '--save-path',
                   os.path.join(temp_folder_path, 'save-folder'))
    assign_to_args(args, '--out', temp_folder_path)
    assign_to_args(args, '--predict-from-cache')
    hard_score_rates = np.linspace(0, 1, 11)
    average_precisions = []
    # Predict for all hard score rates
    for hard_score_rate in hard_score_rates:
        logging.info(f'Predicting for hard_score_rate {hard_score_rate}...')
        assign_to_args(args, '--hard_score_rate', hard_score_rate)
        ap_all, _, _, _ = predict(args)
        average_precisions.append(ap_all)
    clear_output()
    vis_plot(hard_score_rates, average_precisions)
    rm_dir(temp_folder_path)
