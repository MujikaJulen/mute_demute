import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SignDataset
from .model import TimmAutoEncoder


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(dataset_folder: str, batch_size: int, device: torch.device, image_size: int):
    train_ds = SignDataset(dataset_folder, "train", segmentated=True, image_size=image_size)
    val_ds = SignDataset(dataset_folder, "val", segmentated=True, image_size=image_size)

    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, train_ds


def train_autoencoder(
    dataset_folder: str,
    model_name: str,
    output_folder: Path,
    device: torch.device,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    image_size: int = 224,
):
    """Entrena un autoencoder (encoder-decoder) para el modelo timm especificado.

    El objetivo es preentrenar el encoder en modo autodecodificador para que las
    capas convolucionales aprendan una buena representación de las imágenes.
    """

    train_loader, val_loader, train_ds = _make_loaders(dataset_folder, batch_size, device, image_size)

    sample_img, _ = train_ds[0]
    input_shape = tuple(sample_img.shape)

    model = TimmAutoEncoder( model_name="vit_small_patch16_224", pretrained=True, input_shape=(3, 224, 224), mask_ratio=0.4).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_checkpoint_path = output_folder / "autoencoder_best.pth"
    best_encoder_path = output_folder / "encoder_best.pth"

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
       
        model.train()
        train_loss = 0.0
        for inputs, _ in tqdm(train_loader, desc=f"Train [{model_name}] Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device, dtype=torch.float32)
            
            outputs, mask = model(inputs) 
            
            loss_matrix = (outputs - inputs) ** 2 
            
            loss = (loss_matrix * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                
                outputs, mask = model(inputs)
                
                loss_matrix = (outputs - inputs) ** 2
                loss = (loss_matrix * mask).sum() / (mask.sum() + 1e-8)
                
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            torch.save(model.encoder.state_dict(), best_encoder_path)

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"[{model_name}] Epoch {epoch+1}/{num_epochs} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} (best={best_val_loss:.4f})"
            )

    print(f"Pretraining completed for {model_name}. Best val loss {best_val_loss:.4f}")
    print(f"Saved best autoencoder: {best_checkpoint_path}")
    print(f"Saved best encoder: {best_encoder_path}")

    config_path = output_folder / "pretrain_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "input_shape": input_shape,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Pretrain encoder-decoder (autoencoder) using the dataset.")
    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Path to the dataset folder (must contain train/ and val/ subfolders).",
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder base for saving pretraining checkpoints. Defaults to outs/pretrain/<model>.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device where to run training.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of pretraining epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for pretraining.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for pretraining.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Spatial size (H,W) to resize segmentated images to (e.g., 224 for ViT).",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable resizing / scaling of images (keeps original resolution).",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    model_name = "vit_small_patch16_224"
    output_folder = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parents[1] / "outs" / "pretrain" / model_name
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nPretraining autoencoder for model: {model_name}")
    train_autoencoder(
        dataset_folder=args.dataset,
        model_name=model_name,
        output_folder=output_folder,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=None if args.no_resize else args.image_size,
    )


if __name__ == "__main__":
    main()
