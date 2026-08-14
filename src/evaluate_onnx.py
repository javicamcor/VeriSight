import os
import glob
import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

# Importamos el dataset que creamos antes para reutilizar el preprocesamiento
from train_cnn_twostream import TwoStreamDeepfakeDataset

def evaluate(data_dir, onnx_path, output_dir):
    print(f"Cargando modelo ONNX desde: {onnx_path}")
    
    # Forzamos CPU Execution para máxima compatibilidad si se corre en local
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Obtener los nombres exactos de las entradas del modelo ONNX
    input_spatial_name = session.get_inputs()[0].name
    input_freq_name = session.get_inputs()[1].name
    
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_images = sorted(glob.glob(os.path.join(real_dir, '*.*')))
    fake_images = sorted(glob.glob(os.path.join(fake_dir, '*.*')))
    
    all_images = real_images + fake_images
    all_labels = [0] * len(real_images) + [1] * len(fake_images)
    
    if len(all_images) == 0:
        print(f"Error: No se encontraron imágenes en {data_dir}")
        return
    
    # IMPORTANTE: Recreamos EXACTAMENTE el mismo conjunto de validación (20%) que usó el entrenamiento
    _, val_paths, _, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Evaluando sobre {len(val_paths)} imágenes de validación...")
    
    # Usamos el DataLoader original para aprovechar el multiprocesamiento
    val_dataset = TwoStreamDeepfakeDataset(val_paths, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    y_true = []
    y_pred = []
    y_scores = []
    
    for inputs_spatial, inputs_freq, labels in tqdm(val_loader, desc="Inferencia ONNX"):
        # Convertir tensores de PyTorch a arrays de Numpy para ONNX Runtime
        spatial_np = inputs_spatial.numpy()
        freq_np = inputs_freq.numpy()
        
        # Ejecutar modelo
        outputs = session.run(None, {
            input_spatial_name: spatial_np,
            input_freq_name: freq_np
        })
        
        # El modelo devuelve logits, pasamos por función sigmoide manual
        logits = outputs[0]
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        
        y_true.extend(labels.numpy().flatten().tolist())
        y_pred.extend(preds.flatten().tolist())
        y_scores.extend(probs.flatten().tolist())
        
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN (Métricas Avanzadas)")
    print("="*50)
    # Target 0: Real, Target 1: Fake
    print(classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)']))
    
    try:
        roc = roc_auc_score(y_true, y_scores)
        print(f"ROC-AUC Score: {roc:.4f}")
    except ValueError:
        print("No se pudo calcular ROC-AUC.")
        
    # Dibujar Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real (0)', 'Fake (1)'], yticklabels=['Real (0)', 'Fake (1)'])
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.title('Matriz de Confusión - Red Two-Stream')
    
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'matriz_confusion.png')
    plt.savefig(cm_path)
    print(f"\n¡Listo! Gráfico de Matriz de Confusión guardado en: {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluar modelo ONNX Two-Stream")
    parser.add_argument('--data_dir', type=str, default='data', help="Carpeta base con subcarpetas 'real' y 'fake'")
    # Apuntamos por defecto a la ruta correcta que me indicaste
    parser.add_argument('--onnx_path', type=str, default='extension_twostream/onnx_model/model_twostream.onnx', help="Ruta al modelo ONNX")
    parser.add_argument('--output_dir', type=str, default='evaluacion_resultados', help="Carpeta donde guardar la gráfica")
    
    args = parser.parse_args()
    evaluate(args.data_dir, args.onnx_path, args.output_dir)
