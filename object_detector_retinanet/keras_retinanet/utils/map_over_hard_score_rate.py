import os
import sys
import argparse
import numpy as np

from object_detector_retinanet.keras_retinanet.bin.predict import main as predict
from object_detector_retinanet.keras_retinanet.utils.logger import configure_logging
from object_detector_retinanet.utils import create_dirpath_if_not_exist, rm_dir


def assign_to_args(args, flag, val):
    try:
        args[args.index(flag) + 1] = val
    except ValueError as e:
        # Case flag doesn't exist, append it
        args.append(flag)
        args.append(val)

if __name__ == '__main__':
    configure_logging()
    args = sys.argv[1:]

    # Inject temp_folder to all temp created files, so we can delete it in the end
    temp_folder_path = './temp'
    create_dirpath_if_not_exist(temp_folder_path)
    assign_to_args(args, '--save-path', os.path.join(temp_folder_path, 'save-folder'))
    assign_to_args(args, '--out_dir', temp_folder_path)
    hard_score_rates = np.linspace(0, 1, 11)
    # Predict for all hard score rates
    for hard_score_rate in hard_score_rates:
        assign_to_args(args, '--hard_score_rate', hard_score_rate)
        ap_all, _, _, _ = predict(args)
    rm_dir(temp_folder_path)

    
