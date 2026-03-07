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
import torchvision.utils as vutils
from tqdm import tqdm
from .dataset import SignDataset


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


def train_model(output_folder: Path, device: torch.device, trained_model, model_img_size):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Aplicamos transfor que tengan sentido a las imagenes -> 
    # Para que nuestro modelo las tenga en cuenta
        # 1º Flip -> Da igual mano derecha que izquierda
        # 2º Giros -> Movimientos ligeros de muñeca
        # 3º Ilumincación -> Cambios en la iluminación
        # 4º Tamaño -> Tamaño adecuado para cada modelo

        # Normalización -> ImageNet (Modelos entrenados con este dataset)
        
        #Nota -> Resize ya está hecho -> Dataloader -> Segmentation
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((model_img_size, model_img_size))
    ])

    #Las variaciones solo se aplican al Dataset de Train, NO de Val
    val_transforms = transforms.Compose([
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((model_img_size, model_img_size))
    ])

    # Cargamos las direcciones de las imagenes y sus labels -> Train / Test / Val
    dataset_train = SignDataset("./dataset_filtered/dataset", "train",image_size=model_img_size, transforms=train_transforms)
    dataset_val = SignDataset("./dataset_filtered/dataset", "val", image_size=model_img_size, transforms=val_transforms)

    # Create DataLoaders for the datasets
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, pin_memory=pin_memory,
                              num_workers=4, persistent_workers = True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, pin_memory=pin_memory,
                             num_workers=4, persistent_workers = True)

    # Define the model, loss function, and optimizer
    # model = ConvolutionalNeuralNetwork(len(dataset_train.all_classes)).to(device)
    model = timm.create_model(trained_model, pretrained=True, num_classes=len(dataset_train.all_classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Training loop with validation and saving best weights
    num_epochs = 12
    best_val_loss = float("inf")
    output_folder = Path(output_folder)
    best_model_path = (
        output_folder / "best_model_gait.pth"
    ) 


    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        i=0
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            #Guardamos el primer batch de cada modelo
            # Verificar -> 
                # 1º - Las imágenes se cargan bien.
                # 2º - Se aplican las transformaciones correctamente.
            if i == 0 and epoch == 0: 
                for j in range(10): 
                    img = inputs[j] 
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1) 
                    carpeta_batch = os.path.join(output_folder, "batch_images")
                    os.makedirs(carpeta_batch, exist_ok=True)
                    filename = os.path.join(carpeta_batch, f"img_{j}_transformed.png")
                    vutils.save_image(img, filename)
                print(f"\nPrimer batch del modelo guardado.")
            i = 1
            inputs_cuda = inputs.to(device)
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
                inputs_cuda = inputs.to(device)
                targets_cuda = targets.to(device)
                outputs = model(inputs_cuda)
                loss = criterion(outputs, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    with open("models.json", "r", encoding="utf-8") as m:
        model_data = json.load(m)
    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")

    modelos = model_data["models_to_evaluate"]
    img_size = model_data["image_size"]

    for n_model in range(len(modelos)):
        output_folder = f"./trained_model/model_{modelos[n_model]}"
        # output_folder = os.path.join(output_folder, model)
        os.makedirs(output_folder, exist_ok=True)
        print("Training Timm model with: " + modelos[n_model])
        train_model(output_folder, device=device, trained_model=modelos[n_model], model_img_size=img_size[n_model])

    # Set the seed for reproducibility
    torch.manual_seed(42)
