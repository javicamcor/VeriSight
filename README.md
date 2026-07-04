# VeriSight: Detección de Deepfakes mediante Análisis Espectral

> **VeriSight** es un proyecto interdisciplinar (Matemáticas, Ingeniería Informática y Ciberseguridad) propuesto para los Premios de la Cátedra de Ciberseguridad a la Innovación en Ciberseguridad e IA, a desarrollar en GSEC Málaga.

## Resumen Ejecutivo
La proliferación de imágenes sintéticas generadas por IA ha abierto un nuevo vector de ataque basado en la ingeniería social. **VeriSight** actúa como un escudo proactivo: una extensión de navegador que analiza imágenes en tiempo real. Frente a las "cajas negras" del Deep Learning masivo, esta solución fundamenta su detección en el **análisis matemático de señales en el dominio de la frecuencia**.

## Fundamento Matemático
Las redes neuronales generativas (GANs, modelos de difusión) utilizan operaciones matemáticas (*up-sampling*) que introducen patrones periódicos y ruido de cuadrícula en la estructura de la imagen. Estos artefactos son invisibles para el ojo humano, pero claramente identificables al aplicar la **Transformada Discreta de Fourier 2D (2D DFT)**.

Este repositorio contiene la **Prueba de Concepto (PoC)** algorítmica que valida esta hipótesis, demostrando cómo extraer la firma espectral de una imagen para diferenciar características orgánicas humanas de artefactos sintéticos.

## Estructura de la Prueba de Concepto (PoC)
El script incluido en este repositorio realiza el siguiente flujo de trabajo:
1. **Ingesta:** Carga imágenes reales y generadas por IA.
2. **Preprocesamiento:** Conversión a escala de grises.
3. **Análisis Espectral:** Aplicación de la Transformada Rápida de Fourier (FFT) mediante `numpy`.
4. **Extracción:** Generación de espectrogramas de magnitud y visualización de altas frecuencias.
