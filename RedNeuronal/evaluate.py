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

def evaluate_and_plot(loader, model, dataset_name, output_folder, classes):
    model.eval()
    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_inputs.append(inputs.numpy())
            all_outputs.append(probs.numpy())
            all_targets.append(targets.numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Lets plot the confusion matrix as a heatmap
    indexes_of_outputs = []
    for elements in all_outputs:
        indexes_of_outputs.append(np.argmax(elements))
    indexes_of_outputs = np.array(indexes_of_outputs)

    # Classes -> Dataset correspondiente
    classes = classes.keys()
    # Set the confusion matrix
    map = confusion_matrix(all_targets, indexes_of_outputs)

    # Plot and save the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        map,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )

    plt.xlabel("Predicción")
    plt.ylabel("Objetivo")
    plt.title("Matriz de Confusión Test")
    plt.savefig(output_folder / f"confusion_matrix_{dataset_name}.png")
    plt.show()

    # Let's obtain the accuracy, precision, recall and F1 score for the dataset
    predictions = np.zeros(len(all_targets))
    for i in range(len(all_targets)):
        if all_targets[i] == all_outputs[i].argmax():
            predictions[i] = 1
        else:
            predictions[i] = 0

    # Calculate accuracy
    accuracy = 100 * sum(predictions) / len(predictions)
    print(f"Accuracy: {accuracy:.4f}%")

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, indexes_of_outputs, average=None
    )

    print(f"{'CLASE':<10} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'CANTIDAD (Support)'}")
    print("-" * 60)

    for i in range(10):
        print(
            f"{classes[i]:<10} {prec[i]:.2f}       {rec[i]:.2f}       {f1[i]:.2f}       {support[i]}"
        )

    metrics = {
        "Accuracy": accuracy,
        "Clase": classes,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Cantidad": support,
    }
    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 6 decimal places
    df = df.round(6)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(16, 10))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    with open("models.json", "r", encoding="utf-8") as m:
        model_data = json.load(m)
    device = get_device("auto")
    print(f"Using device: {device}")

    modelos = model_data["models_to_evaluate"]
    img_size = model_data["image_size"]

    for n_model in range(len(modelos)):
        models_folder = f"./trained_model/model_{modelos[n_model]}"
        output_folder = f"./out_model"
        # output_folder = os.path.join(output_folder, model)
        os.makedirs(output_folder, exist_ok=True)
        print("Evaluating model: " + modelos[n_model])

        # Normalizamos los datos de entrada
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Generamos el Dataloader de test -> Validación del modelo
        dataset_test_dir = SignDataset("./dataset", "test", image_size=img_size, transforms=test_transform)

        pin_memory = True if device.type == "cuda" else False
        dataset_test = DataLoader(dataset_test_dir, batch_size=32, shuffle=True, pin_memory=pin_memory,
                              num_workers=4, persistent_workers = True)

        # Load the best model weights
        model = timm.create_model(modelos[n_model], pretrained=True, num_classes=20)
        model.load_state_dict(torch.load(output_folder / "best_model.pth"))

        metrics = {}
        # Evaluate and plot for train, validation and test datasets
        metrics["test"] = evaluate_and_plot(dataset_test, model, "test", output_folder, classes = dataset_test.all_classes, device)

        # save  metrics as csv
        pd.DataFrame(metrics["test"]).to_csv(output_folder / "metrics_test.csv")
        pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

        # Save the metrics as an image
        save_metrics_as_picture(metrics["test"], output_folder / "metrics_test.png")

        print(f"Evaluation on {modelos[n_model]} complete!")

        # Set the seed for reproducibility
        torch.manual_seed(42)