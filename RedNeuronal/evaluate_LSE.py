import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Importamos las clases locales
from .dataset_LSE import LSEDataset
from .model_LSE import LSEClassifier

def evaluate():
    # 1. Configuración de rutas y carpetas
    # Definimos la carpeta de salida "outs"
    current_dir = Path(__file__).parent
    output_folder = current_dir / "outs"
    output_folder.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Cargar dataset de test
    # Es importante usar el mismo archivo CSV generado por create_lse_dataset.py
    test_csv = current_dir / "test_lse.csv"
    if not test_csv.exists():
        print(f"Error: No se encuentra {test_csv}. Ejecuta primero create_lse_dataset.py")
        return

    test_ds = LSEDataset(test_csv)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    class_names = test_ds.classes

    # 3. Cargar el modelo entrenado
    model = LSEClassifier(output_dim=len(class_names)).to(device)
    model_path = current_dir / "models" / "best_lse_model.pth"

    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}. Ejecuta train.py primero.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Pesos del modelo cargados correctamente.")

    # 4. Fase de Inferencia
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    # 5. Cálculo y guardado de métricas (Reporte)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Guardar reporte en CSV
    report_csv_path = output_folder / "metrics_report_lse.csv"
    df_report.to_csv(report_csv_path)
    print(f"Reporte de métricas guardado en: {report_csv_path}")
    print("\n--- Resumen de Clasificación ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 6. Generación y guardado de la Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel("Predicción (Modelo)")
    plt.ylabel("Realidad (Etiqueta)")
    plt.title("Matriz de Confusión - Lenguaje de Señas Español")
    
    # Guardar la imagen en la carpeta outs
    plot_path = output_folder / "confusion_matrix_lse.png"
    plt.savefig(plot_path)
    print(f"Matriz de confusión guardada en: {plot_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    evaluate()