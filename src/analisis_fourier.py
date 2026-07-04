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
    Aplica la 2D DFT y un filtro de paso alto para aislar 
    las altas frecuencias (donde reside el ruido de la IA).
    """
    # =========================================================================
    # 1. MÓDULO DE INGESTA Y PREPROCESAMIENTO
    # =========================================================================
    
    # Cargamos la imagen directamente en escala de grises para eliminar la 
    # información de color y quedarnos con la intensidad lumínica estructural.
    img_espacial = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_espacial is None:
        raise FileNotFoundError(f"No se pudo encontrar o cargar la imagen en: {ruta_imagen}")

    # =========================================================================
    # 2. ACONDICIONAMIENTO DE LA SEÑAL (VENTANA DE HANN)
    # =========================================================================
    
    # Para evitar la "cruz" brillante en el espectro causada por el salto brusco 
    # en los bordes de la imagen, aplicamos una ventana de Hann que difumina 
    # perimetralmente la imagen hacia el negro absoluto de forma progresiva.
    filas, columnas = img_espacial.shape
    ventana_x = np.hanning(columnas)
    ventana_y = np.hanning(filas)
    ventana_2d = np.outer(ventana_y, ventana_x)
    
    img_suavizada = img_espacial * ventana_2d

    # =========================================================================
    # 3. MÓDULO DE EXTRACCIÓN MATEMÁTICA (ANÁLISIS FRECUENCIAL)
    # =========================================================================
    
    # np.fft.fft2 aplica la Transformada Rápida de Fourier en 2 dimensiones.
    f = np.fft.fft2(img_suavizada)
    
    # fftshift mueve el componente de frecuencia cero (iluminación media) al centro.
    fshift = np.fft.fftshift(f)

    # =========================================================================
    # 4. FILTRO DE FEATURES (AISLAMIENTO DE ALTAS FRECUENCIAS)
    # =========================================================================
    
    # Calculamos el centro exacto de la matriz.
    centro_x, centro_y = filas // 2, columnas // 2
    radio_filtro = 30 # Tamaño del círculo central a bloquear (bajas frecuencias)
    
    # Creamos una máscara circular.
    x, y = np.ogrid[:filas, :columnas]
    mascara = (x - centro_x)**2 + (y - centro_y)**2 <= radio_filtro**2
    
    # Aplicamos la máscara: asignamos 1 al centro para que, al aplicar el 
    # logaritmo en el siguiente paso (log(1) = 0), el centro quede completamente oscuro,
    # permitiendo que el ruido sintético de los bordes destaque visualmente.
    fshift[mascara] = 1 

    # Calculamos el espectro de magnitud en escala logarítmica para visualizarlo.
    espectro_magnitud = 20 * np.log(np.abs(fshift) + 1)
    
    return img_espacial, espectro_magnitud

def visualizar_comparativa(img_real, espectro_real, img_ai, espectro_ai):
    """
    Genera un panel (subplot) para comparar visualmente los espectros 
    de frecuencia de la imagen real frente a la sintética.
    """
    plt.figure(figsize=(12, 10))

    # --- FILA 1: ANÁLISIS DE LA IMAGEN REAL ---
    plt.subplot(2, 2, 1)
    plt.imshow(img_real, cmap='gray')
    plt.title('Imagen Real (Humana)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
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
    plt.title('Espectro 2D DFT (Artefactos Sintéticos Revelados)')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle("VeriSight PoC: Aislamiento de Altas Frecuencias", fontsize=16, y=1.02)
    
    # Guardamos la imagen final para adjuntarla al repositorio
    plt.savefig('comparativa_espectros.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # =========================================================================
    # EJECUCIÓN PRINCIPAL DEL SCRIPT
    # =========================================================================
    
    ruta_real = os.path.join('..', 'data', 'real_face.jpg')
    ruta_ai = os.path.join('..', 'data', 'ai_face.jpg')

    try:
        print("[*] Iniciando motor de análisis VeriSight...")
        
        print("[1/3] Extrayendo huella espectral de la imagen real...")
        img_real, esp_real = calcular_espectro_frecuencia(ruta_real)
        
        print("[2/3] Extrayendo huella espectral de la imagen sintética...")
        img_ai, esp_ai = calcular_espectro_frecuencia(ruta_ai)
        
        print("[3/3] Generando cuadro de mando visual...")
        visualizar_comparativa(img_real, esp_real, img_ai, esp_ai)
        
        print("[+] ¡Proceso completado con éxito! Archivo 'comparativa_espectros.png' guardado.")
        
    except Exception as e:
        print(f"[-] Error fatal durante la ejecución: {e}")