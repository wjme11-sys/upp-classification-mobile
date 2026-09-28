"""
preprocess_dataset.py

Script de redimensionamiento y unificación del dataset de úlceras por presión (UPP).

Este script toma las imágenes de múltiples fuentes, las redimensiona a 224x224 píxeles
en formato RGB, y las organiza en una estructura de carpetas por clase (1, 2, 3, 4).

Uso:
    python preprocess_dataset.py --input <carpeta_entrada> --output <carpeta_salida>

Ejemplo:
    python preprocess_dataset.py --input "C:/datasets/Dataset 3" --output "C:/datasets/Dataset_3_resized"

Autor: [Tu nombre]
Fecha: 2025
"""

import os
import argparse
import cv2
import numpy as np


# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_INPUT_DIR = "Dataset 3"
DEFAULT_OUTPUT_DIR = "Dataset_3_resized"
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
TARGET_CHANNELS = 3  # 1 para escala de grises, 3 para RGB
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def preprocess_images(input_base_dir, output_base_dir,
                      target_width=TARGET_WIDTH,
                      target_height=TARGET_HEIGHT,
                      target_channels=TARGET_CHANNELS):
    """
    Redimensiona y unifica las imágenes del dataset.

    Args:
        input_base_dir (str): Ruta de la carpeta raíz con subcarpetas de clase (1, 2, 3, 4).
        output_base_dir (str): Ruta de la carpeta de salida.
        target_width (int): Ancho objetivo en píxeles.
        target_height (int): Alto objetivo en píxeles.
        target_channels (int): Número de canales (1 = gris, 3 = RGB).

    Returns:
        tuple: (processed_count, skipped_count)
    """
    # --- 1. Preparar directorio de salida ---
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"✅ Directorio de salida creado: '{output_base_dir}'")
    else:
        print(f"ℹ️  Directorio de salida ya existe: '{output_base_dir}'")

    processed_count = 0
    skipped_count = 0

    print(f"\n🚀 Iniciando el procesamiento de imágenes de '{input_base_dir}'...\n")

    # --- 2. Recorrer las carpetas de clase (1, 2, 3, 4) ---
    for folder_num in range(1, 5):
        input_folder_path = os.path.join(input_base_dir, str(folder_num))
        output_folder_path = os.path.join(output_base_dir, str(folder_num))

        # Crear subcarpeta de salida si no existe
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"📁 Subdirectorio de salida creado: '{output_folder_path}'")

        # Verificar que exista la carpeta de entrada
        if not os.path.isdir(input_folder_path):
            print(f"⚠️  ADVERTENCIA: La carpeta de entrada '{input_folder_path}' no existe. Saltando.")
            continue

        print(f"\n📂 Procesando imágenes en '{input_folder_path}'...")

        # --- 3. Procesar cada archivo de imagen ---
        for filename in os.listdir(input_folder_path):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                input_file_path = os.path.join(input_folder_path, filename)
                output_file_path = os.path.join(output_folder_path, filename)

                try:
                    # Cargar imagen según el número de canales
                    if target_channels == 3:
                        image = cv2.imread(input_file_path, cv2.IMREAD_COLOR)
                    elif target_channels == 1:
                        image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        print(f"❌ ERROR: target_channels debe ser 1 o 3. Actual: {target_channels}")
                        skipped_count += 1
                        continue

                    if image is None:
                        print(f"⚠️  No se pudo cargar la imagen: {input_file_path}. Saltando.")
                        skipped_count += 1
                        continue

                    # --- 4. Preprocesamiento ---
                    # 4.1. Redimensionar
                    image_resized = cv2.resize(
                        image,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA
                    )

                    # 4.2. Asegurar el número correcto de canales
                    if target_channels == 3 and len(image_resized.shape) == 2:
                        image_final = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
                    elif target_channels == 1 and len(image_resized.shape) == 3:
                        image_final = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    else:
                        image_final = image_resized

                    # 4.3. Eliminar dimensión extra si es monocromática
                    if target_channels == 1 and len(image_final.shape) == 3 and image_final.shape[2] == 1:
                        image_final = image_final.squeeze()

                    # --- 5. Guardar imagen procesada ---
                    cv2.imwrite(output_file_path, image_final)
                    processed_count += 1

                except Exception as e:
                    print(f"❌ ERROR al procesar '{input_file_path}': {e}. Saltando.")
                    skipped_count += 1

    # --- 6. Resumen final ---
    print(f"\n{'='*60}")
    print("  📊 RESUMEN DEL PROCESAMIENTO")
    print(f"{'='*60}")
    print(f"  ✅ Imágenes procesadas exitosamente: {processed_count}")
    print(f"  ⚠️  Imágenes saltadas (con errores): {skipped_count}")
    print(f"  📁 Imágenes redimensionadas guardadas en: '{output_base_dir}'")
    print(f"{'='*60}\n")

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Redimensiona y unifica imágenes de úlceras por presión a 224x224 RGB."
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT_DIR,
        help=f'Carpeta de entrada (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Carpeta de salida (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--width', type=int, default=TARGET_WIDTH,
        help=f'Ancho objetivo (default: {TARGET_WIDTH})'
    )
    parser.add_argument(
        '--height', type=int, default=TARGET_HEIGHT,
        help=f'Alto objetivo (default: {TARGET_HEIGHT})'
    )

    args = parser.parse_args()

    preprocess_images(
        input_base_dir=args.input,
        output_base_dir=args.output,
        target_width=args.width,
        target_height=args.height
    )


if __name__ == "__main__":
    main()