import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class LSEDataset(Dataset):
    def __init__(self, csv_path, label_encoder=None):
        df = pd.read_csv(csv_path)
        
        # Guardamos el encoder para que train y test usen la misma numeración
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(df['label'])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(df['label'])
            
        self.features = df.iloc[:, 1:].values.astype('float32')
        self.classes = self.label_encoder.classes_

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)