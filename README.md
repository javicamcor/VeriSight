# VeriSight: Detección de Deepfakes mediante Análisis Espectral

> **VeriSight** es un proyecto interdisciplinar (Matemáticas, Ingeniería Informática y Ciberseguridad) propuesto para los Premios de la Cátedra de Ciberseguridad a la Innovación en Ciberseguridad e IA, a desarrollar en GSEC Málaga.

## Resumen Ejecutivo
La proliferación de imágenes sintéticas generadas por IA ha abierto un nuevo vector de ataque basado en la ingeniería social. **VeriSight** actúa como un escudo proactivo: una extensión de navegador que analiza imágenes en tiempo real. Frente a las "cajas negras" del Deep Learning masivo, esta solución fundamenta su detección en el **análisis matemático de señales en el dominio de la frecuencia**.

## Fundamento Matemático
Las redes neuronales generativas (GANs, modelos de difusión) utilizan operaciones matemáticas (*up-sampling*) que introducen patrones periódicos y ruido de cuadrícula en la estructura de la imagen. Estos artefactos son invisibles para el ojo humano, pero claramente identificables al aplicar la **Transformada Discreta de Fourier 2D (2D DFT)**.

## Origen de los Datos (Dataset)
Para garantizar el rigor y la reproducibilidad de esta Prueba de Concepto, las imágenes de muestra utilizadas en la carpeta `/data` provienen de fuentes reconocidas:

* **Imagen Real (`real_face.jpg`):** Extraída del *dataset* público [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset), un estándar académico de alta calidad utilizado mundialmente en la investigación de visión por computador.
* **Imagen Sintética (`ai_face.jpg`):** Generada mediante [ThisPersonDoesNotExist], basada en arquitecturas generativas (GANs).

Este repositorio contiene la **Prueba de Concepto (PoC)** algorítmica que valida esta hipótesis, demostrando cómo extraer la firma espectral de una imagen para diferenciar características orgánicas humanas de artefactos sintéticos.

## Estructura de la Prueba de Concepto (PoC)
El script incluido en este repositorio realiza el siguiente flujo de trabajo:
1. **Ingesta:** Carga imágenes reales y generadas por IA.
2. **Preprocesamiento:** Conversión a escala de grises.
3. **Análisis Espectral:** Aplicación de la Transformada Rápida de Fourier (FFT) mediante `numpy`.
4. **Extracción:** Generación de espectrogramas de magnitud y visualización de altas frecuencias.


## Interpretación de los Resultados (Conclusiones de la PoC)

El aislamiento de altas frecuencias revela de forma empírica la diferencia fundamental entre la fotónica real y la generación sintética:

* **Espectro de la Imagen Real (Arriba a la derecha):** La distribución de la energía frecuencial es suave, difusa y aleatoria. Esto se debe a que la luz real interactúa de forma natural con las lentes y los sensores físicos de las cámaras. No existen patrones forzados en las altas frecuencias.
* **Espectro de la Imagen Sintética (Abajo a la derecha):** Se aprecian claramente **líneas geométricas prominentes (trazos horizontales/verticales) y destellos estructurados**. Estos artefactos son la firma matemática del algoritmo. Cuando las redes generativas (GANs o Modelos de Difusión) realizan operaciones de *up-sampling* para crear nuevos píxeles a partir de ruido, introducen involuntariamente un "ruido de cuadrícula" periódico.

**Conclusión:** Mientras que a simple vista ambas imágenes son indistinguibles (dominio espacial), la Transformada Discreta de Fourier demuestra que los rostros sintéticos carecen de la aleatoriedad orgánica a nivel estructural, validando la eficacia de **VeriSight** como método de detección de *Deepfakes*.
