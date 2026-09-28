# Ficha Técnica del Modelo

Detalles técnicos del modelo final de clasificación de Úlceras por Presión (UPP).

---

## Información General

| Campo | Valor |
|---|---|
| **Arquitectura** | EfficientNet-B0 |
| **Framework de entrenamiento** | TensorFlow 2.15+ / Keras |
| **Formato de despliegue** | TensorFlow Lite (`.tflite`) |
| **Tipo de cuantización** | Dynamic Range Quantization (`tf.lite.Optimize.DEFAULT`) |
| **Tamaño final del modelo** | 14.2 MB |
| **Tamaño del modelo Keras original** | ~29 MB |
| **Reducción por cuantización** | ~51% |
| **Número de parámetros** | ~5.3 M |
| **Entrada** | Imagen RGB 224×224 píxeles |
| **Salida** | 4 clases (softmax) |

---

## Clases

El modelo clasifica imágenes en cuatro grados clínicos de UPP, según la clasificación del *European Pressure Ulcer Advisory Panel* (EPUAP):

| Índice | Clase | Descripción clínica |
|---|---|---|
| 0 | Grado I | Eritema que no blanquea |
| 1 | Grado II | Pérdida parcial del espesor de la piel |
| 2 | Grado III | Pérdida total del espesor de la piel |
| 3 | Grado IV | Exposición de hueso, tendón o músculo |

El orden de las clases está definido en `labels.txt` y debe coincidir con el orden utilizado durante el entrenamiento.

---

## Preprocesamiento

El modelo espera imágenes con el siguiente preprocesamiento:

1. **Formato:** RGB
2. **Tamaño:** 224×224 píxeles
3. **Normalización:** Específica de EfficientNet (`preprocess_input`)
   - Primera normalización: `[0, 255] → [0, 1]` (división por 255)
   - Segunda normalización: `[0, 1] → [-1, 1]` (mapeo a rango de EfficientNet)

**En Android (Kotlin):** se replica con dos `NormalizeOp` consecutivas:
```kotlin
ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeMethod.BILINEAR))
    .add(NormalizeOp(0f, 255f))       // [0, 255] → [0, 1]
    .add(NormalizeOp(-1f, 2f))        // [0, 1] → [-1, 1]
    .build()
