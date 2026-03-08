from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import timm
from .dataset import SignDataset
from model import EncoderWithClassifier

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

def evaluate_and_plot(loader, model, dataset_name, output_folder, classes_dict, device):
    model.eval()
    model.to(device) 
    
    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device) 
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_inputs.append(inputs.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    indexes_of_outputs = np.argmax(all_outputs, axis=1) 
    class_names = list(classes_dict.keys())
    
    ### CREAMOS LA CONFUSION MATRIX ###
    cm = confusion_matrix(all_targets, indexes_of_outputs)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicción")
    plt.ylabel("Objetivo")
    plt.title(f"Matriz de Confusión - {dataset_name}")
    plt.savefig(output_folder / f"confusion_matrix_{dataset_name}.png", bbox_inches="tight")
    plt.close()

    ### CALCULAMOS LOS PARÁMETROS ###
    accuracy = 100 * np.mean(all_targets == indexes_of_outputs)
    print(f"\nAccuracy: {accuracy:.4f}%")

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, indexes_of_outputs, average=None
    )

    print(f"\n{'CLASE':<10} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'CANTIDAD'}")
    print("-" * 60)
    for i in range(len(class_names)):
        print(f"{class_names[i]:<10} {prec[i]:.2f}       {rec[i]:.2f}       {f1[i]:.2f}       {support[i]}")

    metrics = {
        "Clase": class_names,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Cantidad": support,
        "Total Acuracy": accuracy
    }
    return metrics, accuracy


### FUNCIÓN QUE GUARDA LOS PARÁMETROS ###
def save_metrics_as_picture(metrics, filepath):
    df = pd.DataFrame(metrics)
    df = df.round(4)

    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 1)) 
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
        
    device = get_device("auto")
    print(f"Using device: {device}")  
    model_name = "vit_small_patch16_224"
    image_model = 224
    models_folder = Path(f"./trained_model/model_{model_name}")
    output_folder = Path(f"./out_model/{model_name}")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating model: {model_name}")

    ### DEFINIMOS EL DATASET Y EL MODELO ###
    test_transform = transforms.Compose([
        transforms.Resize((image_model, image_model)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_test_dir = SignDataset("./dataset_filtered/dataset","test", image_size=224, transforms=test_transform)

    pin_memory = True if device.type == "cuda" else False
    dataset_test = DataLoader(dataset_test_dir, batch_size=32, shuffle=False, pin_memory=pin_memory,
                            num_workers=4, persistent_workers=True)

    model = EncoderWithClassifier(hidden_size=384, num_labels=19, freeze_encoder=True)
    
    weights_path = models_folder / "best_model_gait.pth"
        
    model.load_state_dict(torch.load(weights_path, map_location=device))

    ### LAMAMOS A LA EVALUACION DEL MODELO ###
    metrics = {}
    metrics_dict, acc = evaluate_and_plot(dataset_test, model, "test", output_folder, dataset_test_dir.all_classes, device)
    metrics["test"] = metrics_dict

    # Guardamos en CSV
    pd.DataFrame(metrics["test"]).to_csv(output_folder / "metrics_test.csv", index=False)

    # Guardamos como imagen
    save_metrics_as_picture(metrics["test"], output_folder / "metrics_test.png")
    print(f"Evaluation on {model_name} complete! Test Accuracy: {acc:.2f}%")
    torch.manual_seed(42)