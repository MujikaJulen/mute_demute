import os
import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT_DATASET = Path(__file__).parent.parent / "dataset"
SUBSETS = ["train", "val", "test"]
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Verificación de seguridad antes de empezar
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ERROR: No se encuentra el archivo {MODEL_PATH}. "
                            f"Asegúrate de haberlo descargado y puesto en esa carpeta.")

# Configuración MediaPipe
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH)) # Usamos la ruta absoluta corregida
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def process_subset(subset_name):
    # Se preparan los archivos CSV
    # Tendrán: label, x0, y0, z0, ..., x20, y20, z20 (64 columnas en total)
    subset_path = ROOT_DATASET / subset_name
    output_csv = Path(__file__).parent / f"{subset_name}_lse.csv"
    
    # Header: label, x0, y0, z0 ... x20, y20, z20
    header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        if not subset_path.exists():
            print(f"⚠️ Salto: No existe la carpeta {subset_path}")
            return

        # 2. Recorrer los subdirectorios (train\test\val) y las carpetas de señas (A, B, C...)
        for label in sorted(os.listdir(subset_path)):
            label_dir = subset_path / label
            if not label_dir.is_dir(): 
                continue

            print(f"Procesando {subset_name} -> Etiqueta: {label}")

            for img_name in os.listdir(label_dir):
                img_path = label_dir / img_name
                image = cv2.imread(str(img_path))
                if image is None: 
                    continue
                
                # Convertir para MediaPipe
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, 
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # 3. Detección de manos y extracción de landmarks
                detection_result = detector.detect(mp_image)

                # 4. Extraer puntos si se detectó una mano
                if detection_result.hand_landmarks:
                    hand_points = detection_result.hand_landmarks[0]
                    row = [label]
                    for lm in hand_points:
                        row.extend([lm.x, lm.y, lm.z])
                    writer.writerow(row)
                else:
                    print(f" [!] No se detectó mano en {img_path}, se omitirá.")

    print(f"\n✅ Proceso terminado. Dataset guardado en: {output_csv}")

if __name__ == "__main__":
    for subset in SUBSETS:
        process_subset(subset)
    print("✅ Datasets CSV creados correctamente.")