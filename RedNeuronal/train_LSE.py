from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset_LSE import LSEDataset
from .model_LSE import LSEClassifier

def train_lse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(exist_ok=True)

    # 1. Cargar Datasets
    train_ds = LSEDataset(Path(__file__).parent / "train_lse.csv")
    val_ds = LSEDataset(Path(__file__).parent / "val_lse.csv", label_encoder=train_ds.label_encoder)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Inicializar
    model = LSEClassifier(output_dim=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Scheduler: Reduce LR si el accuracy de val no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, pred = torch.max(model(x), 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        print(f"Época {epoch+1} - Acc: {acc:.4f} - LR: {optimizer.param_groups[0]['lr']}")
        
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_dir / "best_lse_model.pth")

if __name__ == "__main__":
    train_lse()