from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importamos las clases locales
from .dataset_LSE import LSEDataset
from .model_LSE import LSEClassifier

# Número de épocas para el entrenamiento
num_epochs = 5


def get_device(force: str = "auto") -> torch.device:
    """Selecciona el dispositivo (CPU/CUDA) basándose en la disponibilidad."""
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(output_folder: Path, models_folder: Path, device: torch.device):
    # 1. Cargar Datasets
    current_dir = Path(__file__).parent
    train_csv = current_dir / "train_lse.csv"
    val_csv = current_dir / "val_lse.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("No se encuentran los archivos CSV. Ejecuta create_lse_dataset.py primero.")

    dataset_train = LSEDataset(train_csv)
    dataset_val = LSEDataset(val_csv, label_encoder=dataset_train.label_encoder)

    # 2. Configurar DataLoaders
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, pin_memory=pin_memory)

    # 3. Instanciar Modelo, Pérdida y Optimizador
    num_classes = len(dataset_train.classes)
    model = LSEClassifier(input_dim=63, output_dim=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    # 4. Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 5. Variables de control
    best_val_acc = 0.0
    best_model_path = models_folder / f"best_lse_model_{num_epochs}_epochs.pth"

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    # 6. Bucle de Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for inputs, targets in loop:
            inputs_cuda, targets_cuda = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_cuda)
            loss = criterion(outputs, targets_cuda)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # 7. Fase de Validación
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda, targets_cuda = inputs.to(device), targets.to(device)
                outputs = model(inputs_cuda)
                loss = criterion(outputs, targets_cuda)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets_cuda.size(0)
                correct += (predicted == targets_cuda).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        acc = 100 * correct / total
        
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(acc)

        scheduler.step(acc)

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 5 == 0:
            print(f" >> [Epoch {epoch+1}] Loss: T {avg_train_loss:.4f} | V {avg_val_loss:.4f} | Acc: {acc:.2f}%")

    print(f"\nEntrenamiento finalizado. Mejor Acc: {best_val_acc:.2f}%")
    print(f"Modelo guardado en: {best_model_path}")

    # Generamos gráficas en la ruta absoluta de outs
    plot_training_history(history, output_folder)


def plot_training_history(history, output_folder):
    """Genera y guarda las gráficas en el directorio especificado."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 5))

    # Gráfica de Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
    plt.title(f'Historial de Pérdida (Loss) a {len(history["train_loss"])} épocas')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], 'g-', label='Val Accuracy')
    plt.title(f'Historial de Accuracy (%) a {len(history["val_acc"])} épocas')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # GUARDADO EN LA RUTA ESPECÍFICA
    save_path = output_folder / f"loss_and_accuracy_plots_{len(history['train_loss'])}_epochs.png"
    plt.savefig(save_path)
    print(f"Gráficas guardadas en: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 1. Definir directorio base para modelos (relativo al script)
    current_package_dir = Path(__file__).parent
    models_folder = current_package_dir / "models"
    models_folder.mkdir(exist_ok=True, parents=True)

    # 2. DEFINIR RUTA ABSOLUTA PARA LAS GRÁFICAS (Subdirectorio outs solicitado)
    current_dir = Path(__file__).parent
    output_folder = current_dir / "outs"
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Usando dispositivo: {device}")
    
    torch.manual_seed(42)
    
    # Ejecutar entrenamiento
    train_model(output_folder, models_folder, device=device)