from pathlib import Path
import os
import yaml
import pprint
import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch

from source.datasets.ptz_dataset import get_position_datetime_from_labels
from source.track_progress import timefmt

import logging

logger = logging.getLogger(__name__)


class ModelInfo:
    # 1. sort the model by the images
    # 2. train image paths
    # 3. parse the wm, agent, image used to train the model, image capture by the agent
    def __init__(self, model_basedir, model_name, img_dir):
        self.model_name = model_name
        self.model_dir = Path(model_basedir, model_name)
        model_info_path = self.model_dir / "model_info.yaml"
        with open(model_info_path, "r") as f:
            self.info_dict = yaml.safe_load(f)
        self.ori_info_dict = copy.deepcopy(self.info_dict)
        imgnames = np.array(os.listdir(img_dir))
        imgposs, imgtimes = get_position_datetime_from_labels([iname.strip(".jpg") for iname in imgnames])
        # imgtimes = pd.to_datetime(imgtimes, format=timefmt)
        # figure out images that are used in the trainings
        for k in self.info_dict.keys():
            if not k.startswith("restart_") or "start_end" not in self.info_dict[k]["images"]:
                continue
            restart_dict = self.info_dict[k]
            if len(restart_dict["images"]["start_end"]) == 2:
                starttime, endtime = pd.to_datetime(restart_dict["images"]["start_end"], format=timefmt, utc=True)
                idx = np.where((starttime <= imgtimes) & (imgtimes <= endtime))[0]
            else:
                # is an agent and was used for more than once to get images
                idx = []
                timedata = restart_dict["images"]["start_end"]
                for i in range(len(timedata) // 2):
                    # print(timedata[i*2:i*2+1])
                    starttime, endtime = pd.to_datetime(timedata[i*2:i*2+2], format=timefmt, utc=True)
                    idx.append(np.where((starttime <= imgtimes) & (imgtimes <= endtime))[0])
                # print(type(starttime), type(imgtimes))
                idx = np.stack(idx)
            restart_dict["images"]["filename"] = imgnames[idx]

    def __repr__(self):
        from pprint import pformat
        return pformat(self.ori_info_dict)
        # return repr(self.info_dict)

    def get_images_at_restart(self, restart_iter: int):
        if restart_iter > self.info_dict['num_restart']:
            raise ValueError("Current restart of the model is smaller than the input restart iteration, check again")
        return self.info_dict[f"restart_{restart_iter:0>2}"]["images"]["filename"]


class WorldModelInfo(ModelInfo):
    def __init__(self, root_dir, model_name, is_finished=False):
        self.root_dir = Path(root_dir)
        model_type_infer = model_name.split("_")[0]
        assert model_type_infer == "wm", f"Requires a world model (wm_*), but got {model_type_infer}"
        if is_finished:
            basedir = self.root_dir / "finished_models"
        else:
            basedir = self.root_dir / "world_models"
        super().__init__(basedir, model_name, self.root_dir / "collected_imgs")
        self.model_path = self.model_dir / "jepa-latest.pt"


class AgentInfo(ModelInfo):
    def __init__(self, root_dir, model_name, is_finished=False):
        self.root_dir = Path(root_dir)
        model_type_infer = model_name.split("_")[0]
        assert model_type_infer == "ag", f"Requires an agent (ag_*), but got {model_type_infer}"
        if is_finished:
            basedir = self.root_dir / "finished_models"
        else:
            # ag_dir = self.root_dir / "agents"
            basedir = self.root_dir / "agents"
        super().__init__(basedir, model_name, self.root_dir / "collected_imgs")
        self.data_acquire_iteration = []
        self._get_collected_timestamps()
        self.model_target_path = self.model_dir / "jepa-target_latest.pt"
        self.model_policy_path = self.model_dir / "jepa-policy_latest.pt"
        

    def _get_collected_timestamps(self):
        # ! this requires all collected pos, cmd, embed have the same timestamp
        if not (self.root_dir / "collected_positions").exists():
            logger.warning("No timestamps found, cannot analyze collected data info")
            return None
        # positions_at_2024-07-31_17:36:08.173102.txt
        fnames = os.listdir(self.root_dir / "collected_positions")
        ftimes = pd.to_datetime([fn.split("_at_")[-1].strip().strip(".txt") for fn in fnames], format=timefmt, utc=True)
        for k in self.info_dict.keys():
            if not k.startswith("restart_") or "start_end" not in self.info_dict[k]["images"]:
                # not restart or hasn't been used for acquiring images
                continue
            idx = []
            restart_dict = self.info_dict[k]
            if "meta" not in restart_dict.keys():
                restart_dict["meta"] = {}
            timedata = restart_dict["images"]["start_end"]
            for i in range(len(timedata) // 2):
                # print(timedata[i*2:i*2+1])
                starttime, endtime = pd.to_datetime(timedata[i*2:i*2+2], format=timefmt, utc=True)
                idx.append(np.where((starttime <= ftimes) & (ftimes <= endtime))[0])
            # print(type(starttime), type(imgtimes))
            idx = np.stack(idx)
            restart_dict["meta"]["collect_timestamp"] = np.array(ftimes.strftime(timefmt))[idx]
            self.data_acquire_iteration.append(int(k.split("_")[-1]))

    def get_collected_data_at_restart(self, restart_iter: int):
        # pos, cmd, embeds
        if restart_iter > self.info_dict['num_restart']:
            raise ValueError(f"Current restart of the model is smaller than the input restart iteration. Max iteration is {self.info_dict['num_restart']}")
        restart_key = f"restart_{restart_iter:0>2}"
        if restart_iter not in self.data_acquire_iteration:
            raise ValueError(f"{restart_key} does not acquire images, available data acquire restarts are {self.data_acquire_iteration}")
        restart_dict = self.info_dict[f"{restart_key}"]
        collts = restart_dict["meta"]["collect_timestamp"]
        return list(zip(*[(f"positions_at_{ts}.txt", f"commands_at_{ts}.txt", f"embeds_at_{ts}.pt", f"rewards_at_{ts}.pt") for ts in collts.ravel()]))

    def load_collected_data(self, restart_iter):
        fnpos, fncmd, fnembed, fnreward = self.get_collected_data_at_restart(restart_iter)
        embed_dir = self.root_dir / "collected_embeds"
        pos_dir = self.root_dir / "collected_positions"
        cmd_dir = self.root_dir / "collected_commands"
        li_embed = []
        li_pos = []
        li_cmd = []
        li_reward = []
        for i in range(len(fnpos)):
            embed = torch.load(embed_dir / fnembed[i])
            reward = torch.load(embed_dir / fnreward[i])
            with open(pos_dir / fnpos[i], "r") as f:
                pos = np.array([l.strip().split(",") for l in f], dtype=float)
            with open(cmd_dir / fncmd[i], "r") as f:
                cmd = np.array([l.strip().split(",") for l in f], dtype=float)
            li_embed.append(embed)
            li_pos.append(pos)
            li_cmd.append(cmd)
            li_reward.append(reward)
        return li_pos, li_cmd, li_embed, li_reward


class ProgressTracker:
    # steps:
    # 1. train a random world model
    # 2. generate dreams by a random trained world model
    # 3. train a random agent (needs to stick to a single world model)
    # 4. gather images using a random trained agent
    # last line is always the last model name
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.prog_file = self.root_dir / "progress_model_names.txt"
        with open(self.prog_file, "r") as f:
            self.model_names, self.finish_time = list(zip(*[map(str.strip, l.split("@")) for l in f]))
        self.prog_dict = {}
        for m, v in zip(self.model_names, self.finish_time):
            if m.startswith("Start lifelong learning"):
                continue
            if m not in self.prog_dict.keys():
                self.prog_dict[m] = {
                    "num_restart": -1,
                    "finish_times": []
                }
            self.prog_dict[m]["num_restart"] += 1
            self.prog_dict[m]["finish_times"].append(v)
        
    def list_models(self):
        return self.prog_dict

    def __repr__(self):
        from pprint import pformat
        return pformat(self.prog_dict)
    # def get_model


def read_train_loss(fpath):
    """
    Reads the training loss data from a file and returns it as a list of lists.
    
    Args:
        fpath (str): The file path of the data file.
        
    Returns:
        tuple: A tuple containing two elements:
            - all_train (list): A list of lists, where each inner list represents the training loss data for one epoch.
            - header (list): A list of strings representing the header of the data file.
    """
    with open(fpath, "r") as fp:
        all_train = []
        onetrain = []
        header = None
        restart = 0
        for i, ln in enumerate(fp):
            if ln.startswith("epoch"):
                if header is None:
                    header = ln.strip().split(",") + ["restart"]
                    continue
                all_train.append(onetrain)
                onetrain = []
                restart += 1
                continue
            onetrain.append(ln.strip().split(",") + [restart])
        if onetrain:
            all_train.append(onetrain)
    return all_train, header


def flatten(matrix):
    return [item for row in matrix for item in row]


def read_fname_embed_from_h5(fpath):
    fnames = []
    embeds = []
    with h5py.File(fpath, "r") as h5f:
        for k in h5f.keys():
            fnames.append(k)
            embeds.append(h5f[k][:])
    return fnames, embeds


def sort_by_time_from_label(embeds, fnames):
    from source.datasets.ptz_dataset import get_position_datetime_from_labels
    
    pos, time = get_position_datetime_from_labels(fnames)
    idx = np.arange(len(fnames))
    idx = np.argsort(time)
    return np.array(embeds)[idx], np.array(fnames)[idx], np.array(pos)[idx], np.array(time)[idx]
 

def scale_pca_tsne_transform(embeds, pca_components=50):
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    embeds_scaled = StandardScaler().fit_transform(embeds)
    embeds_feat = VarianceThreshold(threshold=0.001).fit_transform(embeds_scaled)
    embeds_pca = PCA(n_components=pca_components, svd_solver='auto').fit_transform(embeds_feat)
    embeds_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50, n_jobs=-1).fit_transform(embeds_pca)
    return embeds_tsne, embeds_pca
