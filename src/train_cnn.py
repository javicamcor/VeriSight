import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import argparse
from tqdm import tqdm
import torchvision.models as models
import cv2
from datasets import load_dataset
import onnx

# =====================================================================
# 1. ARQUITECTURA DEL MODELO (Desde cero)
# =====================================================================
class DeepfakeCNN(nn.Module):
    """
    Arquitectura ResNet-18 configurada para entrenarse DESDE CERO (From Scratch).
    Al no usar pesos preentrenados, la red aprenderá a buscar patrones 
    en frecuencias matemáticas (espectrogramas), no en formas físicas.
    """
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        # Entrenamos desde cero (sin pesos previos)
        self.resnet = models.resnet18(weights=None)
        
        # Adaptamos la primera capa para que acepte 1 solo canal (Blanco y Negro)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Sustituimos la última capa para clasificar probabilidad de Deepfake (1 salida)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.resnet(x)

# =====================================================================
# 2. INGESTA DE DATOS Y EXTRACCIÓN MATEMÁTICA (Streaming)
# =====================================================================
class StreamingDeepfakeDataset(IterableDataset):
    def __init__(self, dataset_name, split="train"):
        super().__init__()
        # streaming=True evita descargar las imágenes al disco duro
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        
    def process_image(self, pil_img):
        # 1. Convertir a matriz numpy
        img = np.array(pil_img)
        
        # 2. Asegurar escala de grises y tamaño estándar
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        
        # 3. Ventana de Hann para difuminar bordes
        filas, columnas = img.shape
        ventana_x = np.hanning(columnas)
        ventana_y = np.hanning(filas)
        ventana_2d = np.outer(ventana_y, ventana_x)
        img_suavizada = img * ventana_2d
        
        # 4. Transformada Discreta de Fourier
        f = np.fft.fft2(img_suavizada)
        fshift = np.fft.fftshift(f)
        
        # 5. Filtro de Paso Alto (Bloquear el centro)
        centro_x, centro_y = filas // 2, columnas // 2
        radio_filtro = 15 
        x, y = np.ogrid[:filas, :columnas]
        mascara = (x - centro_x)**2 + (y - centro_y)**2 <= radio_filtro**2
        fshift[mascara] = 1
        
        # 6. Magnitud Logarítmica
        espectro = 20 * np.log(np.abs(fshift) + 1)
        
        # 7. Normalización Z-Score
        mean = np.mean(espectro)
        std = np.std(espectro)
        if std > 0:
            espectro = (espectro - mean) / std
            
        # 8. Convertir a Tensor PyTorch (Canal, Alto, Ancho)
        tensor = torch.from_numpy(espectro).float().unsqueeze(0)
        return tensor

    def __iter__(self):
        for item in self.dataset:
            try:
                # En Defactify, las columnas se llaman 'Image' y 'Label_A'
                pil_img = item['Image']
                label = item['Label_A']
                
                tensor_espectro = self.process_image(pil_img)
                tensor_label = torch.tensor([label], dtype=torch.float32)
                
                yield tensor_espectro, tensor_label
            except Exception:
                # Si una imagen del stream falla, simplemente pasamos a la siguiente
                continue

# =====================================================================
# 3. BUCLE DE ENTRENAMIENTO Y EXPORTACIÓN A ONNX
# =====================================================================
def train_and_export(output_model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")
    
    print("\nConectando con Hugging Face (Streaming)...")
    dataset_name = "Rajarshi-Roy-research/Defactify_Image_Dataset"
    
    train_dataset = StreamingDeepfakeDataset(dataset_name, split="train")
    val_dataset = StreamingDeepfakeDataset(dataset_name, split="validation")
    
    # En streaming no se puede usar shuffle=True, el flujo dicta el orden
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    
    model = DeepfakeCNN().to(device)
    
    # COCOAI (Defactify) está tremendamente desbalanceado: 16.000 Reales vs 80.000 Fakes.
    # pos_weight = Reales / Fakes = 16000 / 80000 = 0.2
    pos_weight = torch.tensor([16000.0 / 80000.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate estándar para From Scratch
    
    # Reductor de velocidad automático: si la precisión se atasca 2 rondas, baja la velocidad a la mitad
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    epochs = 25 # Subimos a 25 rondas para que el modelo tenga tiempo de afinar
    best_val_acc = 0.0
    
    print("\nIniciando entrenamiento de la CNN por Streaming (PyTorch)...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Iteración de entrenamiento
        # Nota: Al ser IterableDataset, tqdm no sabrá el total exacto hasta terminar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Iteración de validación
        model.eval()
        correct = 0
        total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total if total > 0 else 0
        print(f"\nResumen Epoch [{epoch+1}/{epochs}] | Val Accuracy: {val_acc*100:.2f}%")
        
        # Le decimos al reductor de velocidad qué precisión hemos sacado
        # para que decida si es momento de frenar.
        scheduler.step(val_acc)
        
        # Sistema de guardado
        os.makedirs(output_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_model_dir, 'last_model.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f" -> ¡Mejor modelo ({val_acc*100:.2f}%)! Guardando checkpoint...")
            torch.save(model.state_dict(), os.path.join(output_model_dir, 'best_model.pth'))
            
    # Exportación final a Edge AI (ONNX)
    print(f"\nExportando modelo a formato ONNX en {output_model_dir}...")
    best_model_path = os.path.join(output_model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    onnx_path = os.path.join(output_model_dir, 'model.onnx')
    
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=18,          # Versión actualizada para compatibilidad total con PyTorch
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Incrustando pesos internamente para compatibilidad web...")
    merged_model = onnx.load(onnx_path)
    onnx.save_model(merged_model, onnx_path)
    
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        
    print(f"¡Proceso completado! Archivo final: {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar CNN en Streaming para Edge AI")
    parser.add_argument('--output', type=str, default='extension/onnx_model', help="Carpeta de salida")
    
    args = parser.parse_args()
    train_and_export(args.output)