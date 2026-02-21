from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import skimage


class SignDataset(Dataset):
    def __init__(self, dataset_folder):
        self.allimages = []
        self.test=[]
        self.train=[]
        self.val=[]
        for root, dirs, files in os.walk(dataset_folder):

            for image_file in files:

                if image_file.endswith(('.png','.jpg','.jpeg', '.JPG', '.JPEG')):

                    image_path = os.path.join(root, image_file)
                    self.allimages.append((image_path, dirs))
        #IGUAL HAY QUE ALMACENAR ADEMAS DEL DIRS EL ROOT PARA FILTRAR Y TENER UN SOLO OBJETO

                    match root:
                        case "train":
                            self.train.append((image_path, dirs))
                        case "test":
                            self.test.append((image_path, dirs))
                        case "val":
                            self.val.append((image_path, dirs))
                        case _:
                            print("Error: root no coincide con train, test o val")
                    


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): #REVISAR ESTO CÓMO SERÍA PARA QUE FUNCIONE CON LOS CUATRO TIPOS DE IMAGENES DEFINIDAS EN EL DATASET
        image = skimage.io.imread(self.allimages[idx][0])
        label = self.allimages[idx][1]
        return image, label  # Return a dummy label (0) since no labels are available in the dataset

    def plot(self, filepath):
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                img = self.data[i * 10 + j][0].permute(1, 2, 0)
                #change axis 0 and 3
                plt.imshow(img, cmap=plt.cm.binary)
                plt.xlabel(self.data.classes[self.data[i * 10 + j][1]])
        plt.show()
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name 
    output_folder.mkdir(exist_ok=True, parents=True)

    # Data augmentation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )

    dataset_train = CIFAR10Dataset("./data", train=True, transform=transform)
    dataset_test = CIFAR10Dataset("./data", train=False, transform=transform)
    print(f"Dataset length: {len(dataset_train)}")
    print(f"First item: {dataset_train[0]}")
    dataset_train.plot(output_folder / "plot_dataset_example.png")
