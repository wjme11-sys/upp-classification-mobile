"""
evaluate_external.py

Script de validación externa del modelo TFLite de clasificación de UPP.

Este script carga el modelo TFLite optimizado y evalúa su desempeño sobre
imágenes externas al dataset de entrenamiento, obteniendo clase predicha,
confianza y distribución de probabilidades.

Uso:
    python evaluate_external.py --model <ruta_modelo.tflite> --images <carpeta_imagenes>

Ejemplo:
    python evaluate_external.py --model model/efficientnet_ulceras_opt_v2.tflite --images imagenes_externas/

Autor: [Tu nombre]
Fecha: 2025
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_MODEL_PATH = "model/efficientnet_ulceras_opt_v2.tflite"
DEFAULT_IMAGES_DIR = "imagenes_externas"
CLASS_NAMES = ['Grado I', 'Grado II', 'Grado III', 'Grado IV']
IMG_SIZE = (224, 224)


def run_tflite_inference(tflite_model_path, image_path, class_names):
    """
    Ejecuta inferencia TFLite sobre una imagen.

    Args:
        tflite_model_path (str): Ruta al archivo .tflite.
        image_path (str): Ruta a la imagen a evaluar.
        class_names (list): Lista con los nombres de las clases.

    Returns:
        tuple: (clase_predicha, confianza, probabilidades) o (None, None, None) si falla.
    """
    try:
        # 1. Cargar el intérprete TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 2. Cargar, redimensionar y normalizar la imagen
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Preprocesamiento específico de EfficientNetB0
        preprocessed_img = preprocess_input(img_array)

        # 4. Ejecutar inferencia
        interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 5. Post-procesamiento
        probabilities = tf.nn.softmax(output_data[0]).numpy()
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        return predicted_class, confidence, probabilities

    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró la imagen: {image_path}")
        return None, None, None
    except Exception as e:
        print(f"❌ ERROR durante la inferencia: {e}")
        return None, None, None


def evaluate_folder(model_path, folder_path, class_names=CLASS_NAMES):
    """
    Evalúa todas las imágenes en una carpeta.

    Args:
        model_path (str): Ruta al modelo .tflite.
        folder_path (str): Ruta a la carpeta con imágenes externas.
        class_names (list): Lista con los nombres de las clases.

    Returns:
        list: Lista de diccionarios con los resultados.
    """
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: La carpeta '{folder_path}' no existe.")
        return []

    # Filtrar archivos de imagen
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if not image_files:
        print(f"⚠️  No se encontraron imágenes en '{folder_path}'.")
        return []

    print(f"\n{'='*60}")
    print(f"  🔍 PRUEBAS CON IMÁGENES EXTERNAS ({len(image_files)} imágenes)")
    print(f"{'='*60}")

    results = []

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        predicted_class, confidence, probabilities = run_tflite_inference(
            model_path, image_path, class_names
        )

        if predicted_class is not None:
            print(f"\n📷 Imagen: {filename}")
            print(f"   Clase predicha: {predicted_class}")
            print(f"   Confianza: {confidence:.2f}%")
            print(f"   Distribución completa:")
            for i, prob in enumerate(probabilities):
                print(f"      {class_names[i]}: {prob*100:.2f}%")

            results.append({
                'archivo': filename,
                'clase_predicha': predicted_class,
                'confianza': confidence,
                'probabilidades': probabilities.tolist()
            })

    # Resumen final
    if results:
        print(f"\n{'='*60}")
        print("  📊 RESUMEN DE PRUEBAS EXTERNAS")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['archivo']:40} → {r['clase_predicha']:12} ({r['confianza']:.2f}%)")

        conf_promedio = np.mean([r['confianza'] for r in results])
        conf_max = max([r['confianza'] for r in results])
        conf_min = min([r['confianza'] for r in results])

        print(f"\n  Confianza promedio: {conf_promedio:.2f}%")
        print(f"  Confianza máxima:   {conf_max:.2f}%")
        print(f"  Confianza mínima:   {conf_min:.2f}%")
        print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validación externa de un modelo TFLite de clasificación de UPP."
    )
    parser.add_argument(
        '--model', '-m',
        default=DEFAULT_MODEL_PATH,
        help=f'Ruta al modelo .tflite (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--images', '-i',
        default=DEFAULT_IMAGES_DIR,
        help=f'Carpeta con imágenes externas (default: {DEFAULT_IMAGES_DIR})'
    )

    args = parser.parse_args()

    evaluate_folder(args.model, args.images)


if __name__ == "__main__":
    main()