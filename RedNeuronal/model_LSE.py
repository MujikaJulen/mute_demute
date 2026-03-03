import torch.nn as nn

class LSEClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, output_dim=26):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)