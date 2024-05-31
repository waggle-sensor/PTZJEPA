import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image 
from torch.utils.data import Dataset
from pathlib import Path

class PTZImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, target_transform=None):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'{img_dir} is not a directory')
        self.img_dir = Path(img_dir)
        if annotations_file is None:
            self.img_labels = [im.stem for im in self.img_dir.glob('*.jpg')]
        else:
            # the filename has only one column
            self.img_labels = list(pd.read_csv(annotations_file, header=None)[0])
        self.transform = transform
        self.target_transform = target_transform

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
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


