import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import random
import onnx
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. ARQUITECTURA DEL MODELO (Two-Stream CNN)
# =====================================================================
class TwoStreamDeepfakeCNN(nn.Module):
    """
    Arquitectura de Dos Vías (Two-Stream):
    - Vía 1 (Espacial): EfficientNet-B0, recibe imagen RGB (3 canales).
    - Vía 2 (Frecuencial): EfficientNet-B0 modificada para recibir espectro FFT (1 canal).
    - Fusión: Concatena las 1280 características de cada vía y clasifica.
    """
    def __init__(self):
        super(TwoStreamDeepfakeCNN, self).__init__()
        
        # --- Vía Espacial (RGB) ---
        self.spatial_stream = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.spatial_stream.classifier = nn.Identity() # Quitamos la capa final, salida de 1280
        
        # --- Vía Frecuencial (Fourier) ---
        self.freq_stream = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Adaptamos la primera capa para 1 solo canal
        original_conv1 = self.freq_stream.features[0][0]
        self.freq_stream.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            self.freq_stream.features[0][0].weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
        self.freq_stream.classifier = nn.Identity() # Quitamos la capa final, salida de 1280
        
        # --- Dropouts Independientes para forzar preferencia ---
        # Penalizamos un poco más la vía espacial (0.4) que la frecuencial (0.2)
        self.dropout_spatial = nn.Dropout(0.4)
        self.dropout_freq = nn.Dropout(0.2)
        
        # --- Fusión y Clasificación Final ---
        # 1280 (Espacial) + 1280 (Frecuencial) = 2560 características
        self.fusion = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(512, 1)
        )

    def forward(self, x_spatial, x_freq):
        out_spatial = self.spatial_stream(x_spatial)
        out_spatial = self.dropout_spatial(out_spatial)
        
        out_freq = self.freq_stream(x_freq)
        out_freq = self.dropout_freq(out_freq)
        
        # Ajustamos matemáticamente los pesos: 40% Espacial, 60% Frecuencial
        out_spatial = out_spatial * 0.4
        out_freq = out_freq * 0.6
        
        # Concatenar las características de ambas vías
        fused_features = torch.cat((out_spatial, out_freq), dim=1)
        
        return self.fusion(fused_features)

# =====================================================================
# 2. INGESTA DE DATOS (Two-Stream)
# =====================================================================
class TwoStreamDeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        
        # Precomputar Ventana de Hann (224x224)
        ventana_x = np.hanning(224)
        ventana_y = np.hanning(224)
        self.ventana_2d = np.outer(ventana_y, ventana_x)
        
        # Precomputar Máscara del Filtro de Paso Alto
        centro_x, centro_y = 112, 112
        radio_filtro = 10 
        x, y = np.ogrid[:224, :224]
        self.mascara = (x - centro_x)**2 + (y - centro_y)**2 <= radio_filtro**2
        
        # Transformación para la imagen RGB (Normalización estándar de ImageNet)
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Leer en color (BGR para OpenCV)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # Fallback en caso de que la imagen esté corrupta
        if img_bgr is None:
            return torch.zeros((3, 224, 224)), torch.zeros((1, 224, 224)), torch.tensor([label], dtype=torch.float32)
        
        # 2. Simulador de Redes Sociales (Compresión JPEG)
        calidad_jpeg = random.randint(60, 95)
        _, img_comprimida = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), calidad_jpeg])
        img_bgr = cv2.imdecode(img_comprimida, cv2.IMREAD_COLOR)
        
        # 3. Extracción de Parche Puro (Random Crop 224x224)
        h, w = img_bgr.shape[:2]
        if h < 224 or w < 224:
            pad_h = max(0, 224 - h)
            pad_w = max(0, 224 - w)
            img_bgr = cv2.copyMakeBorder(img_bgr, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_REFLECT)
            h, w = img_bgr.shape[:2]
            
        start_y = random.randint(0, h - 224)
        start_x = random.randint(0, w - 224)
        img_bgr = img_bgr[start_y:start_y+224, start_x:start_x+224]
        
        # 4. Data Augmentation Extremo
        # 4.1 Simulador de Baja Resolución (Destrucción Aleatoria)
        if random.random() > 0.6:
            scale = random.uniform(0.15, 0.5) # Encoger hasta un 15% (33x33 px)
            small = cv2.resize(img_bgr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            img_bgr = cv2.resize(small, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 4.2 Volteos aleatorios
        if random.random() > 0.5:
            img_bgr = cv2.flip(img_bgr, 1) # horizontal
        if random.random() > 0.5:
            img_bgr = cv2.flip(img_bgr, 0) # vertical
            
        # 4.3 Desenfoque Gaussiano (Blur)
        if random.random() > 0.3:
            ksize = random.choice([3, 5, 7])
            img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)
            
        # 4.4 Brillo y Contraste Aleatorios
        if random.random() > 0.3:
            alpha = random.uniform(0.7, 1.3) # Contraste
            beta = random.randint(-30, 30)   # Brillo
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
            
        # 4.5 Ruido Gaussiano
        if random.random() > 0.5:
            row, col, ch = img_bgr.shape
            sigma = random.randint(5, 20)
            gauss = np.random.normal(0, sigma, (row, col, ch))
            img_bgr = np.clip(img_bgr + gauss, 0, 255).astype(np.uint8)
            
        # --- PROCESAMIENTO VÍA ESPACIAL ---
        # Mantener color real RGB para que la red vea fallos de textura y color
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor_spatial = self.spatial_transform(img_rgb)
        
        # --- PROCESAMIENTO VÍA FRECUENCIAL ---
        # Usamos la misma imagen en escala de grises para Fourier
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        
        # Ventana de Hann (Usar precomputada)
        img_suavizada = img_gray * self.ventana_2d
        
        # Transformada de Fourier y Filtro
        f = np.fft.fft2(img_suavizada)
        fshift = np.fft.fftshift(f)
        fshift[self.mascara] = 1 # Paso alto
        
        # Magnitud Logarítmica y Normalización
        espectro = 20 * np.log(np.abs(fshift) + 1)
        mean = np.mean(espectro)
        std = np.std(espectro)
        if std > 0:
            espectro = (espectro - mean) / std
            
        tensor_freq = torch.from_numpy(espectro).float().unsqueeze(0)
        tensor_label = torch.tensor([label], dtype=torch.float32)
        
        return tensor_spatial, tensor_freq, tensor_label

# =====================================================================
# 3. BUCLE DE ENTRENAMIENTO Y EXPORTACIÓN A ONNX
# =====================================================================
def train_and_export(data_dir, output_model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")
    print(f"Iniciando entrenamiento Two-Stream (Espacial + Frecuencia)")
    
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    # IMPORTANTE: Ordenar los archivos (sorted) asegura que siempre esten en el mismo orden.
    # Así, el random_state=42 del train_test_split siempre separará exactamente las mismas imágenes.
    real_images = sorted(glob.glob(os.path.join(real_dir, '*.*')))
    fake_images = sorted(glob.glob(os.path.join(fake_dir, '*.*')))
    
    num_reals = len(real_images)
    num_fakes = len(fake_images)
    all_images = real_images + fake_images
    all_labels = [0] * num_reals + [1] * num_fakes
    
    if len(all_images) == 0:
        print("No se encontraron imágenes.")
        return
        
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_dataset = TwoStreamDeepfakeDataset(train_paths, train_labels)
    val_dataset = TwoStreamDeepfakeDataset(val_paths, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = TwoStreamDeepfakeCNN().to(device)
    
    weight_val = num_reals / num_fakes if num_fakes > 0 else 1.0
    pos_weight = torch.tensor([weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 15
    best_val_acc = -1.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs_spatial, inputs_freq, labels in progress_bar:
            inputs_spatial = inputs_spatial.to(device)
            inputs_freq = inputs_freq.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_spatial, inputs_freq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        correct = 0
        total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs_spatial, inputs_freq, labels in val_bar:
                inputs_spatial = inputs_spatial.to(device)
                inputs_freq = inputs_freq.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs_spatial, inputs_freq)
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total if total > 0 else 0
        print(f"\nResumen Epoch [{epoch+1}/{epochs}] | Val Accuracy: {val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        os.makedirs(output_model_dir, exist_ok=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            print(f" -> ¡Mejor modelo ({val_acc*100:.2f}%)! Guardando checkpoint...")
            torch.save(model.state_dict(), os.path.join(output_model_dir, 'best_model.pth'))
            
    # Exportación ONNX
    print(f"\nExportando modelo Two-Stream a formato ONNX en {output_model_dir}...")
    best_model_path = os.path.join(output_model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    # Necesitamos dos tensores de prueba (Espacial y Frecuencial) para ONNX
    dummy_spatial = torch.randn(1, 3, 224, 224).to(device)
    dummy_freq = torch.randn(1, 1, 224, 224).to(device)
    onnx_path = os.path.join(output_model_dir, 'model_twostream.onnx')
    
    model.eval()
    torch.onnx.export(
        model, 
        (dummy_spatial, dummy_freq), 
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_spatial', 'input_frequency'],
        output_names=['output'],
        dynamic_axes={
            'input_spatial': {0: 'batch_size'}, 
            'input_frequency': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    print("Incrustando pesos internamente para compatibilidad web...")
    merged_model = onnx.load(onnx_path)
    onnx.save_model(merged_model, onnx_path)
    
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        
    print(f"¡Proceso completado! Archivo final: {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar CNN Two-Stream (Espacial + Frecuencial)")
    parser.add_argument('--data_dir', type=str, default='data', help="Carpeta base con subcarpetas 'real' y 'fake'")
    # Apuntamos la salida por defecto a la nueva carpeta
    parser.add_argument('--output', type=str, default='extension_twostream/onnx_model', help="Carpeta de salida para el modelo ONNX")
    
    args = parser.parse_args()
    train_and_export(args.data_dir, args.output)
