import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image 
from torch.utils.data import Dataset

class DreamDataset(Dataset):
    def __init__(self, dream_dir):
        self.dream_dir = dream_dir
        self.dream_paths = []
        for subdir, dirs, files in os.walk(dream_dir):
            for ffile in files:
                #print(os.path.join(subdir, ffile))
                self.dream_paths.append(os.path.join(subdir, ffile))

    def __len__(self):
        return len(self.dream_paths)

    def __getitem__(self, idx):
        dream_path = self.img_labels[idx]
        dream = torch.open(dream_path)
        return dream


