### EN ESTE SCRIPT SE VA A FILTAR EL TAMAÑO DE LAS IMÁGENES MEDIANTE TÉCNICAS DE SISTEMAS DE PERCEPCIÓN ####
### OBJETIVO -> REDUCIR EL TAMAÑO DE LAS IMAGENES PARA QUE SEAN MÁS FÁCILES DE PROCESAR POR LA RED NEURONAL ###
import cv2
import os
import numpy as np
import skimage
import tqdm 
from skimage.morphology import disk
import matplotlib.pyplot as plt

def filter_images_by_size(dataset_folder, output_folder, scale_size=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(dataset_folder):

        for image_file in files:

            if image_file.endswith(('.png','.jpg','.jpeg', '.JPG', '.JPEG')):

                image_path = os.path.join(root, image_file)
                image = skimage.io.imread(image_path)

                scale_factor = 0.05
                img_prescaled = skimage.transform.rescale(image, scale=scale_factor, channel_axis=2)

                img_hsv = skimage.color.rgb2hsv(img_prescaled)
                img_sat = img_hsv[:,:,1]

                best_threshold = skimage.filters.threshold_otsu(img_sat)
                original_height, original_width = img_sat.shape
                mask = img_sat > best_threshold
                mask = skimage.morphology.opening(mask, disk(3))
                mask = skimage.morphology.closing(mask, disk(3))
                mask = skimage.morphology.dilation(mask, disk(8)) # Para que no recorte al borde justo
                labeled_img = skimage.measure.label(mask)
                regions = skimage.measure.regionprops(labeled_img)
                region = max(regions, key=lambda r: r.area) #Coge la región más grande, que debería ser la que contiene la mano, no entiendo muy bien como va r y por que es una variable local si yo no la he definido. REVISAR
                min_row, min_col, max_row, max_col = region.bbox
                min_row = int(min_row / scale_factor)
                min_col = int(min_col / scale_factor)
                max_row = int(max_row / scale_factor)
                max_col = int(max_col / scale_factor)

                if (max_row - min_row) > (max_col - min_col): # me quedo con el mínimo cuadrado posible
                    image_segmented = image[min_row:max_row, min_col:min_col + (max_row - min_row)]
                else:
                    image_segmented = image[min_row:min_row + (max_col - min_col), min_col:max_col]
                
                img_scaled = skimage.transform.resize(image_segmented, (scale_size, scale_size))
                img_scaled_uint8 = skimage.img_as_ubyte(img_scaled)  # Convierte a uint8

                output_path = os.path.join(output_folder, image_file)  # o un nuevo nombre
                skimage.io.imsave(output_path, img_scaled_uint8)

                # plt.imshow(img_scaled_uint8)
                # plt.show()



if __name__ == "__main__":
    dataset_folder = "dataset"  # Carpeta con las imágenes originales
    output_folder = "filtered_dataset"  # Carpeta para guardar las imágenes filtradas

    filter_images_by_size(dataset_folder, output_folder)