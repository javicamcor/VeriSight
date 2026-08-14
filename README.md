# VeriSight: Detección de Deepfakes en Tiempo Real

> **VeriSight** es una extensión de navegador (disponible para Google Chrome) diseñada para actuar como un escudo proactivo contra la manipulación visual. Analiza imágenes y fotografías en tiempo real mientras navegas, protegiéndote de campañas de desinformación e ingeniería social basadas en imágenes sintéticas (Deepfakes).

## Resumen del Proyecto y Tecnología
La proliferación de imágenes generadas por IA ha abierto nuevos vectores de ataque. Frente a las "cajas negras" del Deep Learning tradicional, VeriSight fundamenta su precisión combinando el **análisis matemático en el dominio de la frecuencia** con una arquitectura avanzada de redes neuronales (**Two-Stream CNN**) para cazar la firma intrínseca que dejan los algoritmos generativos.

> ⚠️ **Enfoque del Modelo:** VeriSight está estrictamente entrenado para analizar **rostros humanos**, no paisajes ni elementos fotográficos cualesquiera. Dado que el objetivo principal de la herramienta es combatir la suplantación de identidad y los *deepfakes* en redes sociales, el entrenamiento se ha priorizado y optimizado exclusivamente con rostros para garantizar la máxima eficacia.

### Arquitectura Two-Stream CNN
El núcleo de la extensión se basa en un modelo preentrenado de última generación modificado para procesar dos flujos de información complementarios:

1. **Vía Espacial (RGB):** Analiza la imagen original para detectar inconsistencias visuales, artefactos en los bordes, texturas antinaturales y anomalías cromáticas.
2. **Vía Frecuencial (Espectro FFT):** Es la pieza clave del proyecto. Las redes generativas (GANs y Modelos de Difusión) utilizan procesos de *up-sampling* para generar resoluciones altas a partir de ruido latente. Esto introduce inevitablemente un "ruido de cuadrícula" periódico imperceptible al ojo humano. Al aplicar la **Transformada Discreta de Fourier 2D (2D DFT)**, estos artefactos sintéticos se revelan de manera evidente. Esta segunda vía de la red se alimenta exclusivamente del espectro de magnitud para identificar esta "huella digital" algorítmica.

Ambas vías se procesan simultáneamente y sus características se concatenan para que el modelo tome una decisión unificada y altamente precisa.

### Características del entrenamiento
Entrenar modelos de detección de deepfakes suele sufrir de *Identity Leakage*. Para lograr un modelo realmente robusto en entornos reales, se aplica:
* **Data Augmentation Extremo:** Desenfoque severo, ruido gaussiano, compresión JPEG destructiva y alteraciones fuertes de color. Obligamos a la red a no fiarse de los rasgos faciales y a depender de las texturas orgánicas y las anomalías frecuenciales.
* **Gran Volumen de Datos:** Se ensambló un dataset amplio y estrictamente balanceado para garantizar la máxima generalización frente a múltiples IAs generativas.

## Conjuntos de Datos (Datasets) Utilizados
Para garantizar el rigor, la variedad y mitigar los sesgos geográficos o tecnológicos, este proyecto entrena sus modelos unificando y balanceando imágenes de los siguientes datasets:
* [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
* [Generated.photos](https://generated.photos)
* [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026)
* [CelebA (Rostros reales)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [bitmind/ArtiFact](https://huggingface.co/datasets/bitmind/ArtiFact)

---

## Cómo Usar la Extensión VeriSight

El uso de la extensión en el navegador ha sido diseñado para ser extremadamente sencillo e intrínsecamente preciso:

1. **Seleccionar la Imagen:** Haz clic derecho sobre cualquier foto o imagen dudosa mientras navegas.
2. **Analizar:** En el menú contextual que aparece, selecciona la opción *"Analizar con VeriSight"*.
3. **Marcar la zona:** La pantalla se oscurecerá y te pedirá que hagas clic izquierdo para marcar la zona exacta del rostro que deseas analizar. La extensión capturará **única y estrictamente una cuadrícula de 224x224 píxeles reales**.

---

## Instalación y Configuración del Entorno de Desarrollo

Si deseas entrenar el modelo o ejecutar pruebas de validación localmente, el proyecto está preparado para ejecutarse en cualquier máquina moderna con Python.

1. **Clonar el repositorio y preparar el entorno:**
   ```bash
   git clone <tu-repositorio>
   cd VeriSight
   python -m venv venv
   source venv/bin/activate  # En Windows usa: venv\Scripts\activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Incluye PyTorch, TorchVision, ONNX, OpenCV, Numpy y las librerías de análisis correspondientes).*

---

## Uso y Entrenamiento del Sistema

El flujo de trabajo de *Machine Learning* se divide en dos scripts principales dentro de la carpeta `src/`.

### 1. Entrenamiento del Modelo (`train_cnn_twostream.py`)
Para iniciar el entrenamiento de la arquitectura de dos vías (incluyendo la generación en caliente de los espectros de Fourier):
```bash
python src/train_cnn_twostream.py
```
> **Nota:** El script buscará las imágenes en las carpetas `data/real/` y `data/fake/`, donde se haya el dataset completo. Al finalizar, exportará los mejores pesos del modelo (`best_model.pth`) y las gráficas de pérdida a la raíz del proyecto.

### 2. Exportación y Evaluación Rigurosa (`evaluate_onnx.py`)
Para integrar el modelo en la extensión de Chrome de manera eficiente, el modelo en PyTorch se exporta al formato estándar **ONNX**. Luego, para simular su rendimiento final, se evalúa utilizando `onnxruntime`:
```bash
python src/evaluate_onnx.py
```
> **Nota:** El script reconstruirá automáticamente el conjunto de validación aislado (20% de los datos) usando la misma semilla del entrenamiento, probará la inferencia y guardará la matriz de confusión.

---

## Resultados y Rendimiento Analítico

El modelo Two-Stream ha demostrado ser extremadamente resiliente, superando obstáculos clásicos de visión por computador y logrando unas métricas excepcionales de generalización.

📊 **Para consultar las métricas exactas (F1-Score, Accuracy Global), la matriz de confusión detallada y las conclusiones sobre el comportamiento matemático del modelo, dirígete al documento de [Reporte de Evaluación de Resultados](evaluacion_resultados/reporte_evaluacion.md).**

---

## Autor

**Javier Campos Córcoles** - *Creador y Desarrollador Principal*
Proyecto desarrollado como investigación independiente