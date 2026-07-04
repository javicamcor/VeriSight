import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calcular_espectro_frecuencia(ruta_imagen):
    """
    Aplica la Transformada Discreta de Fourier 2D (2D DFT) a una imagen.
    Retorna la imagen original en grises y su espectro de magnitud.
    """
    # 1. Módulo de Ingesta y Preprocesamiento
    # La imagen se transforma a escala de grises y se normaliza
    img_espacial = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if img_espacial is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

    # 2. Módulo de Extracción Matemática
    # Aplicación de la Transformada Discreta de Fourier 2D (2D DFT)
    f = np.fft.fft2(img_espacial)
    
    # Desplazamiento del componente de frecuencia cero (DC) al centro del espectro
    fshift = np.fft.fftshift(f)
    
    # Cálculo del espectro de magnitud mediante escala logarítmica para visualización
    # Se suma 1 para evitar log(0)
    espectro_magnitud = 20 * np.log(np.abs(fshift) + 1)
    
    return img_espacial, espectro_magnitud

def visualizar_comparativa(img_real, espectro_real, img_ai, espectro_ai):
    """
    Genera un subplot para comparar visualmente los espectros de frecuencia.
    """
    plt.figure(figsize=(12, 10))

    # Fila 1: Imagen Real
    plt.subplot(2, 2, 1)
    plt.imshow(img_real, cmap='gray')
    plt.title('Imagen Real (Humana)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(espectro_real, cmap='magma')
    plt.title('Espectro 2D DFT (Distribución Orgánica)')
    plt.axis('off')

    # Fila 2: Imagen Sintética (Deepfake)
    plt.subplot(2, 2, 3)
    plt.imshow(img_ai, cmap='gray')
    plt.title('Imagen Sintética (Generada por IA)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(espectro_ai, cmap='magma')
    plt.title('Espectro 2D DFT (Artefactos / Alta Frecuencia)')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle("VeriSight PoC: Análisis Matemático de Señales", fontsize=16, y=1.02)
    
    # Guardar el resultado para adjuntarlo al README
    plt.savefig('comparativa_espectros.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Rutas relativas a las imágenes de prueba
    ruta_real = os.path.join('..', 'data', 'real_face.jpg')
    ruta_ai = os.path.join('..', 'data', 'ai_face.jpg')

    try:
        print("Procesando imagen real...")
        img_real, esp_real = calcular_espectro_frecuencia(ruta_real)
        
        print("Procesando imagen sintética...")
        img_ai, esp_ai = calcular_espectro_frecuencia(ruta_ai)
        
        print("Generando visualización comparativa...")
        visualizar_comparativa(img_real, esp_real, img_ai, esp_ai)
        
        print("¡Proceso completado! Imagen de resultados generada.")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")