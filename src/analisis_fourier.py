"""
VeriSight - Prueba de Concepto (PoC)
------------------------------------
Detección de Deepfakes mediante Análisis Espectral (2D DFT)

Este script procesa imágenes reales y generadas por IA para extraer
su firma matemática en el dominio de la frecuencia, evidenciando las
diferencias estructurales producidas por las redes generativas (GANs).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calcular_espectro_frecuencia(ruta_imagen):
    """
    Aplica la Transformada Discreta de Fourier 2D (2D DFT) a una imagen.
    Retorna la imagen original preprocesada y su espectro de magnitud.
    """
    # =========================================================================
    # 1. MÓDULO DE INGESTA Y PREPROCESAMIENTO
    # =========================================================================
    
    # Cargamos la imagen directamente en escala de grises (cv2.IMREAD_GRAYSCALE).
    # Esto elimina la información de color (canales RGB) y nos deja solo con la 
    # intensidad lumínica estructural, reduciendo el coste computacional.
    img_espacial = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # Verificamos que la imagen se ha cargado correctamente
    if img_espacial is None:
        raise FileNotFoundError(f"No se pudo encontrar o cargar la imagen en: {ruta_imagen}")

    # =========================================================================
    # 2. MÓDULO DE EXTRACCIÓN MATEMÁTICA (ANÁLISIS FRECUENCIAL)
    # =========================================================================
    
    # np.fft.fft2 aplica la Transformada Rápida de Fourier en 2 dimensiones.
    # Convierte la matriz de píxeles (dominio espacial) a una matriz de 
    # números complejos (dominio de la frecuencia).
    f = np.fft.fft2(img_espacial)
    
    # Por defecto, la FFT pone la frecuencia cero (el componente de iluminación 
    # media de la imagen) en la esquina superior izquierda. 
    # fftshift mueve esta frecuencia cero al centro exacto de la matriz 
    # para que el espectrograma sea simétrico y visualmente interpretable.
    fshift = np.fft.fftshift(f)
    
    # =========================================================================
    # 3. FILTRO DE CARACTERÍSTICAS (CÁLCULO DEL ESPECTRO DE MAGNITUD)
    # =========================================================================
    
    # Los valores de la matriz compleja fshift tienen un rango enorme.
    # Para poder visualizar las frecuencias altas (donde se esconde el ruido 
    # de las IAs), calculamos el valor absoluto y aplicamos una transformación 
    # logarítmica (multiplicada por 20, como en decibelios).
    # Sumamos 1 al valor absoluto antes del logaritmo para evitar hacer log(0).
    espectro_magnitud = 20 * np.log(np.abs(fshift) + 1)
    
    return img_espacial, espectro_magnitud

def visualizar_comparativa(img_real, espectro_real, img_ai, espectro_ai):
    """
    Genera un panel (subplot) para comparar visualmente los espectros 
    de frecuencia de la imagen real frente a la sintética.
    """
    # Creamos un lienzo de 12x10 pulgadas
    plt.figure(figsize=(12, 10))

    # --- FILA 1: ANÁLISIS DE LA IMAGEN REAL ---
    plt.subplot(2, 2, 1) # (filas, columnas, posición)
    plt.imshow(img_real, cmap='gray')
    plt.title('Imagen Real (Humana)')
    plt.axis('off') # Ocultamos los ejes X e Y

    plt.subplot(2, 2, 2)
    # Utilizamos el mapa de color 'magma' que resalta muy bien las diferencias
    # de intensidad en los espectrogramas.
    plt.imshow(espectro_real, cmap='magma')
    plt.title('Espectro 2D DFT (Distribución Orgánica)')
    plt.axis('off')

    # --- FILA 2: ANÁLISIS DE LA IMAGEN DE INTELIGENCIA ARTIFICIAL ---
    plt.subplot(2, 2, 3)
    plt.imshow(img_ai, cmap='gray')
    plt.title('Imagen Sintética (Generada por IA)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(espectro_ai, cmap='magma')
    plt.title('Espectro 2D DFT (Artefactos / Alta Frecuencia)')
    plt.axis('off')

    # Ajustamos los espacios entre los subgráficos para que no se superpongan
    plt.tight_layout()
    plt.suptitle("VeriSight PoC: Análisis Matemático de Señales", fontsize=16, y=1.02)
    
    # Guardamos el resultado como una imagen de alta resolución (300 dpi) 
    # en la carpeta del script. Esta es la imagen que subiremos al README de GitHub.
    plt.savefig('comparativa_espectros.png', dpi=300, bbox_inches='tight')
    
    # Mostramos la ventana interactiva en pantalla
    plt.show()

if __name__ == "__main__":
    # =========================================================================
    # EJECUCIÓN PRINCIPAL DEL SCRIPT
    # =========================================================================
    
    # Definimos las rutas relativas a las imágenes guardadas en la carpeta 'data'.
    # os.path.join asegura que las rutas funcionen tanto en Windows como en Mac/Linux.
    ruta_real = os.path.join('..', 'data', 'real_face.jpg')
    ruta_ai = os.path.join('..', 'data', 'ai_face.jpg')

    try:
        print("[*] Iniciando motor de análisis VeriSight...")
        
        # Procesamos la fotografía del dataset FFHQ
        print("[1/3] Extrayendo huella espectral de la imagen real...")
        img_real, esp_real = calcular_espectro_frecuencia(ruta_real)
        
        # Procesamos la fotografía generada por ThisPersonDoesNotExist
        print("[2/3] Extrayendo huella espectral de la imagen sintética...")
        img_ai, esp_ai = calcular_espectro_frecuencia(ruta_ai)
        
        # Generamos la visualización
        print("[3/3] Generando cuadro de mando visual...")
        visualizar_comparativa(img_real, esp_real, img_ai, esp_ai)
        
        print("[+] ¡Proceso completado con éxito! Archivo 'comparativa_espectros.png' guardado.")
        
    except Exception as e:
        # En caso de que falten las imágenes o haya otro fallo, mostramos el error
        print(f"[-] Error fatal durante la ejecución: {e}")