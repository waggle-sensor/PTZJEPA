import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image 
from torch.utils.data import Dataset

class PTZImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels[idx,0] + '.jpg')
        img_path = os.path.join(self.img_dir, self.img_labels[0][idx] + '.jpg')
        image = Image.open(img_path)
        #image = read_image(img_path)
        #label = self.img_labels[idx,0]
        label = self.img_labels[0][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


