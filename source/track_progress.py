# this file includes functionality to track the progress of the lifelong
# learning process.
import datetime
import logging
from typing import Union
import yaml

import numpy as np

from source.datasets.ptz_dataset import get_position_datetime_from_labels
from source.prepare_dataset import get_dirs


logger = logging.getLogger(__name__)
timefmt = "%Y-%m-%d_%H:%M:%S.%f"
persis_dir, coll_dir, tmp_dir = get_dirs()
wm_dir = persis_dir / "world_models"
ag_dir = persis_dir / "agents"

# model name convention:
# {model_type}_{iteration:0>2}_{num:0>2}
# wm_00_20 (world model, 00 th generation, 20 th model)
# ag_12_53 (agent, 12 th generation, 53 th model)

# Concept
# 1. epoch: image set for current training (eg. 20 movement 30 iteration = 600)
# 2. restart: model restarts from the previous plateau, multiple epochs inside plateau
# 3. generation: model restarted after N times and it was dropped. A new model was born
# 4. model number: random seed or a id for configuration
# random images -> [WM -> dreams -> agent] -> new images -> [WM -> dreams -> agent] -> new images ...


def initialize_model_info(model_name: str, overwrite: bool = False):
    """Initializes the model information and saves it to a YAML file.

    Args:
        model_name (str): The name of the model.

    Returns:
        pathlib.Path: The path to the model directory.
    """
    model_type, model_gen, model_id = model_name.split("_")
    model_gen = int(model_gen)
    model_id = int(model_id)
    model_parent_dir = wm_dir if model_type == "wm" else ag_dir
    model_dir = model_parent_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing world model at %s", model_dir)
    info_dict = {
        "model_name": model_name,
        "model_type": model_type,
        "model_gen": model_gen,
        "model_id": model_id,
        "num_restart": -1,
    }
    if (model_dir / "model_info.yaml").exists() and not overwrite:
        raise RuntimeError(f"Model info already exists at {model_dir}")
    with open(model_dir / "model_info.yaml", "w") as f:
        yaml.dump(info_dict, f)
    return model_dir


# Need to save world model details
# including:
# model id, model restart iteration, training data timestamp, number of images, training epochs
# Time is in "%Y-%m-%d_%H:%M:%S.%f" format
def save_model_info(
    model_name: str,
    parent_model_name: Union[str, None],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    num_epoch: int,
):
    """Save model information to a YAML file.

    Args:
        model_name (str): The full name of the model.
        parent_model_name (Union[str, None]): The name of the parent model. None if it's the first run.
        start_time (datetime.datetime): The start time of the training.
        end_time (datetime.datetime): The end time of the training.
        num_epoch (int): The number of epochs used to train model to reach plateau.
    Raises:
        ValueError: if this is not the first time model is run (i.e., the parent model name is None
                    and the restart iteration is not 0.)
    """

    # Model name is the full model name
    # model_xx_yy
    model_type, model_gen, model_id = model_name.split("_")
    model_gen = int(model_gen)
    model_id = int(model_id)
    # model_type = "world_model" if is_world_model else "agent"
    model_parent_dir = wm_dir if model_type == "wm" else ag_dir
    # model_dir = model_parent_dir / model_name
    info_fpath = model_parent_dir / model_name / "model_info.yaml"
    logger.info("Saving %s to %s", model_name, info_fpath)
    labels = [fp.stem for fp in coll_dir.glob("*.jpg")]
    num = len(labels)
    _, datetimes = get_position_datetime_from_labels(labels)
    with open(info_fpath, "r") as f:
        info_dict = yaml.safe_load(f)
    # Check that the model info matches with the model name
    assert (
        info_dict["model_name"] == model_name
    ), f"Model name does not match with the model info! {model_name} != {info_dict['model_name']}"
    assert (
        info_dict["model_gen"] == model_gen
    ), f"Model generation does not match with the model info! {model_gen} != {info_dict['model_gen']}"
    assert (
        info_dict["model_id"] == model_id
    ), f"Model id does not match with the model info! {model_id} != {info_dict['model_id']}"
    assert (
        info_dict["model_type"] == model_type
    ), f"Model type does not match with the model info! {model_type} != {info_dict['model_type']}"
    restart_iter = info_dict["num_restart"] + 1
    # Sort by time will know who is the parent and find out the flow
    # this method won't consider more than one instance running at the same time
    # infer parent model restart iteration
    if parent_model_name is None:
        if restart_iter == 0:
            # this means this is the first model (world model!) to run
            parent_restart_iter = None
        else:
            raise ValueError(
                "Parent model name is required for restarts except the first run"
            )
    else:
        parent_model_parent_dir = (
            wm_dir if parent_model_name.split("_")[0] == "wm" else ag_dir
        )
        with open(
            parent_model_parent_dir / parent_model_name / "model_info.yaml", "r"
        ) as f:
            parent_info = yaml.safe_load(f)
            # Last key is the latest restart iteration
            # parent_restart_iter = int(list(parent_info.keys())[-1].split("_")[1])
            parent_restart_iter = parent_info["num_restart"]
    info_dict[f"restart_{restart_iter:0>2}"] = {
        "parent_model": parent_model_name,
        "parent_model_restart": parent_restart_iter,
        "train": {
            "start_end": [
                start_time.strftime(timefmt),
                end_time.strftime(timefmt),
            ],
            "num_epochs": num_epoch,
        },
        "images": {
            "start_end": [
                np.min(datetimes).strftime(timefmt),
                np.max(datetimes).strftime(timefmt),
            ],
            "num_images": num,
        },
    }
    info_dict["num_restart"] += 1
    with open(info_fpath, "w") as f:
        yaml.dump(info_dict, f)


def update_progress(current_model_name: str):
    # last line is always the last model name
    prog_file = persis_dir / "progress_model_names.txt"
    with open(prog_file, "a") as f:
        f.write(current_model_name + "\n")
    # now update the last model name to the current model
