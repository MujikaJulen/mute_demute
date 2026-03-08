import os

import matplotlib
import skimage.io
import skimage.transform
import torch
from torch.utils.data import Dataset

from RedNeuronal.image_filtering import segmentation


class SignDataset(Dataset):
    def __init__(
        self,
        dataset_folder,
        split="train",
        segmentated=False,
        image_size: int | None = 224,
    ):
        self.samples = []
        split_folder = os.path.join(dataset_folder, split)
        self.all_classes = {}
        self.segmentated = segmentated
        self.image_size = image_size

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
        if self.segmentated:
            image = segmentation(image_path, scale_size=self.image_size).astype("float32")
        else:
            image = skimage.io.imread(image_path).astype("float32")
            # Resize all images to a consistent resolution for the model.
            if self.image_size is not None and (
                image.shape[0] != self.image_size or image.shape[1] != self.image_size
            ):
                image = skimage.transform.resize(
                    image,
                    (self.image_size, self.image_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype("float32")

        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.all_classes[label]
        return image, label


# if __name__ == "__main__":
#    train = SignDataset("./dataset","train")
#    print("Hola")