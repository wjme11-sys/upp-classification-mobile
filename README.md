# Clasificación de Úlceras por Presión en Dispositivos Móviles

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://www.tensorflow.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.15%2B-yellow)](https://www.tensorflow.org/lite)
[![Android](https://img.shields.io/badge/Android-Kotlin-green)](https://developer.android.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema móvil de clasificación de Úlceras por Presión (UPP) basado en **EfficientNet-B0** y **TensorFlow Lite**, con inferencia local en dispositivos Android mediante **Edge Computing**.

Este repositorio acompaña al artículo:
> *"Desarrollo de un clasificador móvil de úlceras por presión con recursos limitados: metodología, resultados iniciales y lecciones aprendidas"*

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Resultados Clave](#-resultados-clave)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Requisitos](#-requisitos)
- [Instalación y Uso](#-instalación-y-uso)
- [Dataset](#-dataset)
- [Resultados Detallados](#-resultados-detallados)
- [Limitaciones](#-limitaciones)
- [Trabajo Futuro](#-trabajo-futuro)
- [Cómo Citar](#-cómo-citar)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

---

## 🎯 Descripción

Este proyecto implementa un sistema completo para clasificar **Úlceras por Presión (UPP)** en cuatro grados clínicos (I–IV) definidos por el *European Pressure Ulcer Advisory Panel* (EPUAP):

- **Grado I:** Eritema que no blanquea
- **Grado II:** Pérdida parcial del espesor de la piel
- **Grado III:** Pérdida total del espesor de la piel
- **Grado IV:** Exposición de hueso, tendón o músculo

### Aportes principales

1. **Pipeline completo y reproducible** — desde la composición del dataset hasta el despliegue en Android, utilizando exclusivamente recursos computacionales gratuitos (Google Colab Free).
2. **Modelo optimizado para móvil** — EfficientNet-B0 convertido a TFLite (14.2 MB) con cuantización dinámica.
3. **Aplicación Android funcional** — inferencia local en menos de 500 ms, sin conexión a internet.
4. **Documentación de lecciones aprendidas** — 9 obstáculos técnicos identificados y sus soluciones.
5. **Código abierto bajo licencia MIT** — para facilitar la replicación y mejora por parte de la comunidad.

---

## 📊 Resultados Clave

| Métrica | Valor |
|---|---|
| **Accuracy en validación** | **47.23%** (307 imágenes) |
| **Mejor val_accuracy durante entrenamiento** | 54.40% (época 22) |
| **Recall Grado I** | **0.99** (72/73) |
| **Recall Grado IV** | **0.70** (53/76) |
| **Validación externa (4 imágenes)** | **3/4 (75%)** |
| **Tiempo de inferencia en móvil** | 320–420 ms |
| **Tamaño del modelo TFLite** | 14.2 MB |
| **Reducción por cuantización** | ~51% (de 29 MB) |

---

## 🏗️ Arquitectura del Sistema

El pipeline se organiza en cinco fases:

```
┌─────────────────────┐
│  1. Dataset         │   1,200 imágenes
│     (5 fuentes)     │   Grados I-IV
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Preprocesamiento│   Redimensionamiento a 224×224
│     + Augmentation  │   Rotación, zoom, brillo, shifts
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Entrenamiento   │   EfficientNet-B0 + Transfer Learning
│     (2 fases)       │   Fase 1: capas congeladas (10 épocas)
│                     │   Fase 2: fine-tuning (20 épocas, LR 5e-6)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Conversión      │   TensorFlow Lite
│     TFLite          │   Cuantización dinámica (14.2 MB)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. App Android     │   Kotlin + CameraX
│     (Edge Computing)│   Inferencia local <500 ms
└─────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
upp-classification-mobile/
├── README.md                          # Este archivo
├── LICENSE                            # Licencia MIT
├── .gitignore                         # Exclusiones
│
├── notebooks/                         # Notebooks de Colab
│   └── Definitivo_eficientNetB0.ipynb # Entrenamiento + evaluación
│
├── model/                             # Modelo entrenado
│   ├── efficientnet_ulceras_opt_v2.tflite
│   ├── labels.txt
│   └── README.md
│
├── android-app/                       # App Android (Kotlin)
│   ├── TfLiteClassifier.kt
│   ├── MainActivity.kt
│   ├── activity_main.xml
│   ├── build.gradle.kts
│   └── README.md
│
├── scripts/                           # Scripts Python auxiliares
│   ├── preprocess_dataset.py
│   ├── evaluate_external.py
│   ├── requirements.txt
│   └── README.md
│
├── docs/                              # Documentación y figuras
│   ├── confusion_matrix_efficientnet.png
│   ├── confusion_matrix_mobilenetv2.png
│   └── README.md
│
└── dataset/                           # Instrucciones del dataset
    └── README.md
```

---

## ⚙️ Requisitos

### Entrenamiento (Python)
- Python 3.10–3.13
- TensorFlow 2.15+ (o 2.16+ para Python 3.13)
- OpenCV, Pillow, scikit-learn, matplotlib, seaborn

Instalación:
```bash
cd scripts/
pip install -r requirements.txt
```

### Despliegue (Android)
- Android Studio Hedgehog o superior
- Kotlin 1.9+
- Android SDK 24+
- TensorFlow Lite 2.15+

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/wjme11-sys/upp-classification-mobile.git
cd upp-classification-mobile
```

### 2. Entrenamiento del modelo

Abre el notebook en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

O ejecuta localmente:

```bash
cd scripts/
pip install -r requirements.txt
python preprocess_dataset.py --input "Dataset 3" --output "Dataset_3_resized"
```

Luego abre `notebooks/Definitivo_eficientNetB0.ipynb` y ejecuta todas las celdas.

### 3. Validación externa

```bash
python evaluate_external.py \
    --model model/efficientnet_ulceras_opt_v2.tflite \
    --images imagenes_externas/
```

### 4. Despliegue en Android

1. Copia `model/efficientnet_ulceras_opt_v2.tflite` y `model/labels.txt` a `app/src/main/assets/`
2. Integra los archivos de `android-app/` en tu proyecto
3. Compila y ejecuta en un dispositivo Android

Consulta `android-app/README.md` para instrucciones detalladas.

---

## 📊 Dataset

El dataset final está compuesto por **1,200 imágenes** de UPP (Grados I–IV), provenientes de cinco fuentes:

| Fuente | Tipo | Acceso |
|---|---|---|
| Roboflow Universe | Público | https://universe.roboflow.com |
| Kaggle | Público | https://www.kaggle.com |
| figshare (Chang C. W., 2021) | Público | https://doi.org/10.6084/m9.figshare.17206940.v1 |
| FU Medical AI (PIID) | Público | https://doi.org/10.6084/m9.figshare.17206940.v1 |
| Dataset propio | Privado | Recolectado en hospital |

**Nota:** Las imágenes **NO se incluyen** en este repositorio por derechos de autor. Consulta `dataset/README.md` para instrucciones de composición.

### Imágenes de validación externa

Las 4 imágenes utilizadas en la validación externa fueron obtenidas del **Banco de Imágenes del GNEAUPP**:
https://gneaupp.info/seccion/banco-de-imagenes/

Estas imágenes cuentan con consentimiento explícito para uso científico y divulgativo.

---

## 📈 Resultados Detallados

### Matriz de confusión — EfficientNet-B0

| Real \ Predicha | Grado I | Grado II | Grado III | Grado IV | Total |
|---|---|---|---|---|---|
| **Grado I** | **72** | 1 | 0 | 0 | 73 |
| **Grado II** | 46 | **16** | 10 | 12 | 84 |
| **Grado III** | 14 | 19 | **19** | 22 | 74 |
| **Grado IV** | 2 | 9 | 12 | **53** | 76 |
| **Total** | 134 | 45 | 41 | 87 | 307 |

### Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Grado I | 0.54 | **0.99** | 0.70 | 73 |
| Grado II | 0.36 | 0.19 | 0.25 | 84 |
| Grado III | 0.46 | 0.26 | 0.33 | 74 |
| Grado IV | 0.61 | **0.70** | 0.65 | 76 |
| **Accuracy** | — | — | **0.47** | **307** |

### Validación con imágenes externas

| Imagen | Grado real | Clase predicha | Acierto | Confianza |
|---|---|---|---|---|
| Grado I | Grado I | Grado I | ✅ | 32.26% |
| Grado II | Grado II | Grado II | ✅ | 30.86% |
| Grado III | Grado III | Grado IV | ❌ | 33.29% |
| Grado IV | Grado IV | Grado IV | ✅ | 37.46% |
| **Promedio** | | **3/4 (75%)** | | **33.47%** |

---

## ⚠️ Limitaciones

Este trabajo reconoce las siguientes limitaciones:

1. **Tamaño reducido del dataset:** 1,200 imágenes heterogéneas de cinco fuentes.
2. **Dataset ampliado no procesado:** El dataset de ~13,000 imágenes no pudo entrenarse por limitaciones de GPU gratuita.
3. **Sin validación clínica formal:** Las pruebas con imágenes externas no constituyen validación clínica con especialistas.
4. **Confusión en grados intermedios:** Los grados II y III presentan recalls bajos (0.19 y 0.26).
5. **Sensibilidad a iluminación:** No se han explorado condiciones extremas de captura.
6. **Clasificación vs. segmentación:** No se calcula el área de la úlcera en cm².

---

## 🔮 Trabajo Futuro

1. **Corto plazo:** Procesar el dataset ampliado (~13,000 imágenes) con GPUs dedicadas.
2. **Mediano plazo:** Validación clínica formal con especialistas (kappa de Cohen).
3. **Mediano plazo:** Transición a segmentación semántica (U-Net + EfficientNet).
4. **Largo plazo:** Estudio prospectivo en entorno hospitalario real.
5. **Largo plazo:** Expansión a otras heridas crónicas (úlceras venosas, pie diabético).

---

## 📚 Cómo Citar

Si utilizas este código o modelo en tu investigación, por favor cita:

```bibtex
@article{[apellido]2025upp,
  title={Desarrollo de un clasificador móvil de úlceras por presión con recursos limitados: metodología, resultados iniciales y lecciones aprendidas},
  author={[Apellido, Nombre] and others},
  journal={[Revista]},
  year={2025},
  note={Código disponible en: https://github.com/wjme11-sys/upp-classification-mobile}
}
```

> **Nota:** Esta sección se actualizará una vez el artículo sea publicado.

---

## 📄 Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

- **Autor de correspondencia:** [Williasmjavier Mora Espinosa]
- **Email:** [william.mora2@unipamplona.edu.co]
- **Institución:** [Universisad de Pamplona]
- **GitHub:** [@wjme11-sys](https://github.com/wjme11-sys)

Para consultas sobre el dataset compuesto o colaboraciones, por favor contactar indicando afiliación institucional y propósito de uso.

---

## 🙏 Agradecimientos

- **GNEAUPP** (Grupo Nacional para el Estudio y Asesoramiento en Úlceras por Presión y Heridas Crónicas) por proporcionar imágenes de validación externa con consentimiento para uso científico.
- La comunidad de **TensorFlow** por las herramientas de código abierto.
- **Google Colab** por proporcionar recursos computacionales gratuitos.

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐**

</div>
