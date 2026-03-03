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


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.con2D_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.activation1 = nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con2D_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.activation2 = nn.ReLU()
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Final_Layer3 = nn.Linear(32 * 529, output_dim)
        self.activation3 = nn.Softmax(dim=1)

    def forward(self, x, use_activation=True):
        x1 = self.con2D_1(x)
        x2 = self.activation1(x1)
        x3 = self.max_pooling1(x2)
        x4 = self.con2D_2(x3)
        x5 = self.activation2(x4)
        x6 = self.max_pooling2(x5)
        x7 = self.flatten(x6)
        x8 = self.Final_Layer3(x7)
        x9 = self.activation3(x8)
        return x9
