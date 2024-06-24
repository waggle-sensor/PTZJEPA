import os
from typing import Iterable
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class PTZImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, target_transform=None,
                 return_label=False):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'{img_dir} is not a directory')
        self.img_dir = Path(img_dir)
        if annotations_file is None:
            self.img_labels = [im.stem for im in self.img_dir.glob('*.jpg')]
        else:
            # the filename has only one column and should not have a header
            # the string should not have a suffix
            self.img_labels = list(pd.read_csv(annotations_file, header=None)[0])
        self.transform = transform
        # self.target_transform = target_transform
        self.positions, self.date_times = self._parse_labels()
        # sort the labels by datetime to ensure coherence
        sorted_idx = self.date_times.argsort()
        self.img_labels = self.img_labels[sorted_idx]
        self.positions = self.positions[sorted_idx]
        self.date_times = self.date_times[sorted_idx]
        self.return_label = return_label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels[idx,0] + '.jpg')
        img_path = self.img_dir / (self.img_labels[idx] + '.jpg')
        image = Image.open(img_path)
        #image = read_image(img_path)
        #label = self.img_labels[idx,0]
        label = self.img_labels[idx]
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
        pos, date_time = list(zip(*[label.split('_', maxsplit=1) for label in self.img_labels]))
        positions = [tuple(map(float, p.split(','))) for p in pos]
        date_times = pd.to_datetime(date_time, format="%Y-%m-%d_%H:%M:%S.%f", utc=True)
        return positions, date_times

    def get_position_from_label(self, labels: Iterable[str]):
        positions = []
        if not isinstance(labels, Iterable):
            # coerce to list
            labels = [labels]
        for label in labels:
            poss = label.split('_')[0].split(',')
            positions.append((float(poss[0]), float(poss[1]), float(poss[2])))
        return positions

