import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py


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