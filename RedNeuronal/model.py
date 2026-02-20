import torch
import torch.nn as nn


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnJulen(nn.Module):
    def __init__(self):
        super(CnnJulen, self).__init__()

        self.conv1 = nn.Conv2d(input_channels=3, out_channels=16, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=None)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(input_channels=3, out_channels=32, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=None)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(input_channels=3, out_channels=64, kernel_size=3)
        self.mp3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=None)
        self.relu3 = nn.ReLU()

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(in_features=image_shape, out_features=output_shape)
