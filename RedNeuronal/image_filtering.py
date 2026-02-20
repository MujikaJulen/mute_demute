### EN ESTE SCRIPT SE VA A FILTAR EL TAMAÑO DE LAS IMÁGENES MEDIANTE TÉCNICAS DE SISTEMAS DE PERCEPCIÓN ####
### OBJETIVO -> REDUCIR EL TAMAÑO DE LAS IMAGENES PARA QUE SEAN MÁS FÁCILES DE PROCESAR POR LA RED NEURONAL ###
import cv2
import os
import numpy as np
import skimage
import tqdm 
from skimage.morphology import disk

def filter_images_by_size(dataset_folder, output_folder, min_width, min_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(dataset_folder):
        for type_dataset in os.listdir(dataset_folder, filename):
            for alphabet in os.listdir(os.path.join(dataset_folder, type_dataset)):
                for image_file in os.listdir(os.path.join(dataset_folder, type_dataset, alphabet)):
                    image = skimage.io.imread(os.path.join(dataset_folder, type_dataset, alphabet, image_file))
                    image_gray = skimage.color.rgb2gray(image)
                    best_threshold = skimage.filters.threshold_otsu(image_gray)
                    original_height, original_width = image_gray.shape
                    mask = image_gray < best_threshold
                    mask = skimage.morphology.opening(mask, disk(20))
                    mask = skimage.morphology.closing(mask, disk(20))
                    region = skimage.measure.regionprops(mask)
                    min_row, min_col, max_row, max_col = region.bbox




        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                height, width, _ = image.shape
                if width >= min_width and height >= min_height:
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, image)
                    print(f"Saved: {output_path}")
                else:
                    print(f"Filtered out: {filename} (size: {width}x{height})")
            else:
                print(f"Could not read: {filename}")

if __name__ == "__main__":
    dataset_folder = "dataset"  # Carpeta con las imágenes originales
    output_folder = "filtered_dataset"  # Carpeta para guardar las imágenes filtradas
    min_width = 64  # Ancho mínimo requerido
    min_height = 64  # Alto mínimo requerido

    filter_images_by_size(dataset_folder, output_folder, min_width, min_height)