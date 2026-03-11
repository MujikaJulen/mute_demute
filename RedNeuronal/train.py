import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from .dataset import SignDataset
from .model import EncoderWithClassifier

DEFAULT_IMAGE_SIZE = 224


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


def train_model(output_folder: Path, device: torch.device, trained_model):

    # Cargamos las direcciones de las imagenes y sus labels -> Train / Test / Val
    # Usamos la misma segmentación/resizing que el pretraining para mantener la distribución.
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),      
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = SignDataset("./dataset_filtered/dataset", "train", image_size=DEFAULT_IMAGE_SIZE, transforms=train_transforms)
    dataset_val = SignDataset("./dataset_filtered/dataset", "val", image_size=DEFAULT_IMAGE_SIZE, transforms=val_transforms)

    # Create DataLoaders for the datasets
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, pin_memory=pin_memory, num_workers = 8)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True, pin_memory=pin_memory, num_workers = 8)

    ### ViT Hugging Face ###
    model = EncoderWithClassifier(hidden_size=384, num_labels=19, freeze_encoder= False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Ejemplo de configuración del optimizador
    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())

    #Para entrenar rápidamente la cabeza, manteniendo "mas o menos" los parámetros del encoder -> FINE TUNNING
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},  
        {'params': classifier_params, 'lr': 1e-3} 
    ], weight_decay=1e-4)

    # Training loop with validation and saving best weights
    num_epochs = 100
    patience = 10
    best_val_loss = float("inf")
    output_folder = Path(output_folder)
    best_model_path = (
        output_folder / "best_model_gait.pth"
    ) 
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader):
            # Forward pass
            inputs_cuda = inputs.to(device, dtype=torch.float32)
            targets_cuda = targets.to(device)
            outputs = model(inputs_cuda) 
            loss = criterion(outputs, targets_cuda)
            train_loss += loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device, dtype=torch.float32)
                targets_cuda = targets.to(device)
                outputs = model(inputs_cuda)
                loss = criterion(outputs, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0 
        else:
            patience_counter += 1 

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Comprobación del Early Stopping
        if patience_counter >= patience:
            print(f"\n¡Early stopping activado en la época {epoch + 1}! La validación no ha mejorado en {patience} épocas.")
            break # Rompe el bucle fo

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)
    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "mute_demute/outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")
    model = "vit_small_patch16_224"
    output_folder = f"./model_{model}"
    # output_folder = os.path.join(output_folder, model)
    os.makedirs(output_folder, exist_ok=True)
    print("Training Timm model with: " + model)
    train_model(output_folder, device=device, trained_model=model)