import os

import matplotlib
import skimage.io
import skimage.transform
import torch
from torch.utils.data import Dataset

from RedNeuronal.image_filtering import segmentation


class SignDataset(Dataset):
    def __init__(self, dataset_folder, split="train"):
        self.samples = []
        split_folder = os.path.join(dataset_folder, split)
        self.all_classes = {}

        for ABC in os.listdir(split_folder):
            ABC_path = os.path.join(split_folder, ABC)
            for image_file in os.listdir(ABC_path):
                if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(ABC_path, image_file)
                    label = ABC
                    if label not in self.all_classes.keys():
                        self.all_classes[label] = len(self.all_classes.keys())
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = segmentation(image_path).astype("float32")
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.all_classes[label]
        return image, label


# if __name__ == "__main__":
#    train = SignDataset("./dataset","train")
#    print("Hola")
