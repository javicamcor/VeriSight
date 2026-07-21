import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

def process_fourier(image_path, target_size=(128, 128), radius=30):
    """
    Replica matemáticamente exacta de la Transformada de Fourier implementada
    en el sandbox.js de la extensión (Edge AI).
    """
    # 1. Leer imagen en escala de grises
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        return None
        
    # 2. Redimensionar para homogeneizar el dataset para la CNN
    src = cv2.resize(src, target_size)
    
    # 3. DFT óptimo
    rows, cols = src.shape
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(src, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=0)
    
    # 4. Transformada Discreta de Fourier Compleja
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)
    cv2.dft(complexI, complexI, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # 5. fftshift (Mover bajas frecuencias al centro)
    complexI = np.fft.fftshift(complexI, axes=[0, 1])
    
    # 6. Filtro de Paso Alto (High-Pass Filter) circular
    cx, cy = m // 2, n // 2
    y, x = np.ogrid[-cy:m-cy, -cx:n-cx]
    mask = x*x + y*y <= radius*radius
    complexI[mask] = 0
    
    # 7. Magnitud y Escala Logarítmica
    planes = cv2.split(complexI)
    mag = cv2.magnitude(planes[0], planes[1])
    
    mag += 1
    mag = np.log(mag)
    
    # 8. Normalización de 0 a 255 (vital para las CNN)
    cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
    mag = np.uint8(mag)
    
    return mag

def build_dataset(real_dir, fake_dir, output_dir, target_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    
    X = []
    y = []
    
    # Procesar imágenes REALES (Label 0)
    print("Procesando imágenes reales...")
    real_images = glob.glob(os.path.join(real_dir, '*.*'))
    for img_path in tqdm(real_images, desc="Real (0)"):
        mag = process_fourier(img_path, target_size)
        if mag is not None:
            X.append(mag)
            y.append(0)
            
    # Procesar imágenes FAKE (Label 1)
    print("\nProcesando imágenes deepfake...")
    fake_images = glob.glob(os.path.join(fake_dir, '*.*'))
    for img_path in tqdm(fake_images, desc="Fake (1)"):
        mag = process_fourier(img_path, target_size)
        if mag is not None:
            X.append(mag)
            y.append(1)
            
    # Convertir a Tensores NumPy y guardar
    X = np.array(X)
    y = np.array(y)
    
    # Añadir la dimensión de canal para Keras (N, H, W, Channels)
    X = np.expand_dims(X, axis=-1)
    
    print(f"\nDataset Finalizado: {len(X)} muestras. Shape: {X.shape}")
    
    np.save(os.path.join(output_dir, 'X_data.npy'), X)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y)
    print(f"Archivos guardados en {output_dir}/ (X_data.npy, y_labels.npy)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construir dataset de espectros para la CNN")
    parser.add_argument('--real', type=str, required=True, help="Carpeta con imágenes reales")
    parser.add_argument('--fake', type=str, required=True, help="Carpeta con imágenes generadas por IA")
    parser.add_argument('--output', type=str, default='../data/processed_dataset', help="Directorio de salida para los .npy")
    
    args = parser.parse_args()
    build_dataset(args.real, args.fake, args.output)
