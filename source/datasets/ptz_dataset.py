import os
from typing import List, Union
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class PTZImageDataset(Dataset):
    def __init__(
        self,
        img_dir,
        annotations_file=None,
        transform=None,
        target_transform=None,
        return_label=False,
    ):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"{img_dir} is not a directory")
        self.img_dir = Path(img_dir)
        if annotations_file is None:
            self.img_labels = [im.stem for im in self.img_dir.glob("*.jpg")]
        else:
            # the filename has only one column and should not have a header
            # the string should not have a suffix
            self.img_labels = list(pd.read_csv(annotations_file, header=None)[0])
        self.transform = transform
        # self.target_transform = target_transform
        self.positions, self.date_times = self._parse_labels()
        # sort the labels by datetime to ensure coherence
        sorted_idx = self.date_times.argsort()
        np.random.shuffle(sorted_idx)
        self.img_labels[:] = [self.img_labels[i] for i in sorted_idx]
        self.positions = self.positions[sorted_idx]
        self.date_times = self.date_times[sorted_idx]
        self.return_label = return_label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels[idx,0] + '.jpg')
        img_path = self.img_dir / (self.img_labels[idx] + ".jpg")
        image = Image.open(img_path)
        # image = read_image(img_path)
        label = self.img_labels[idx]
        # Adding a fourth channel as the depth
        # image = torch.cat((image, torch.zeros_like(image[0])), dim=0) 
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        if self.return_label:
            return image, label
        return image, self.positions[idx]

    def _parse_labels(self):
        # Note the suffix should have already been removed at this point
        # 99.99,-92.39,232.0_2024-06-21_04:51:47.291323 (.jpg)
        # position_date_time
        return get_position_datetime_from_labels(self.img_labels)


def get_position_datetime_from_labels(labels: Union[List, str]):
    if isinstance(labels, str):
        # coerce to list
        labels = [labels]
    pos, date_time = list(
        zip(*[label.split("_", maxsplit=1) for label in labels])
    )
    positions = np.array([tuple(map(float, p.split(","))) for p in pos])
    date_times = pd.to_datetime(date_time, format="%Y-%m-%d_%H:%M:%S.%f", utc=True)
    return positions, date_times
