import json
import math
import os
import cv2
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm


def generate_stereo_zip(data_dir: Path, output_dir: Path) -> None:
    """
    Helper function to generate the result zip file for the argoverse stereo challenge.
    Args:
        data_dir: Path to the directory containing the disparity predictions.
        output_dir: Path to the output directory to store the output zip file.
    """

    output_dir.mkdir(exist_ok=True, parents=True)

    num_test_logs = 15
    num_pred_logs = len([path for path in data_dir.iterdir() if path.is_dir()])

    assert (
        num_test_logs == num_pred_logs
    ), f"ERROR: Found {num_pred_logs} logs in the input dir {data_dir}. It must have {num_test_logs}."

    for log_path in tqdm(list(data_dir.iterdir())):
        if not log_path.is_dir():
            continue

        disparity_map_fpaths = log_path.glob("*.png")
        for disparity_map_fpath in disparity_map_fpaths:
            disparity_map_pred = cv2.imread(str(disparity_map_fpath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            assert disparity_map_pred.shape == (
                2056,
                2464,
            ), f"ERROR: The predicted disparity map should be of shape (2056, 2464) but got {disparity_map_pred.shape}."

            assert (
                disparity_map_pred.dtype == "uint16"
            ), f"ERROR: The predicted disparity map should be of type uint16 but got {disparity_map_pred.dtype}."

    report_fpath = data_dir / "model_analysis_report.txt"
    if not report_fpath.is_file():
        print(f"ERROR: Report file {report_fpath} not found! Please add it to the input folder.")

    print("Creating zip file for submission...")
    shutil.make_archive(str(output_dir / "stereo_output"), "zip", data_dir)

    print(f"Zip file ({output_dir}/stereo_output.zip) created succesfully. Please submit it to EvalAI for evaluation.")