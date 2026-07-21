import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

class DeepfakeCNN(nn.Module):
    """
    Arquitectura Convolucional equivalente a la que teníamos en Keras.
    PyTorch es mucho más estable y nos permitirá exportar a ONNX sin bugs.
    """

    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        # Bloque 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # Output: 64x64
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # Output: 32x32
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) # Output: 16x16
        
        self.flatten = nn.Flatten()
        
        # 64 canales * 16 de ancho * 16 de alto
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Capa de salida
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def train_and_export(dataset_dir, output_model_dir):
    print("Cargando dataset...")
    X_path = os.path.join(dataset_dir, 'X_data.npy')
    y_path = os.path.join(dataset_dir, 'y_labels.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error crítico: No se encontraron los archivos .npy en {dataset_dir}")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    # Keras usa (N, H, W, C) pero PyTorch usa (N, C, H, W).
    # Como dataset_builder genera (N, 128, 128, 1), lo permutamos:
    X = np.transpose(X, (0, 3, 1, 2))
    
    # Normalizar imágenes al rango [0, 1]
    X = X.astype('float32') / 255.0
    y = y.astype('float32').reshape(-1, 1)
    
    # Barajar
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Entrenando con {len(X_train)} muestras y validando con {len(X_val)}")
    
    # Crear Tensores y DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Inicializar modelo, pérdida y optimizador
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")
    
    model = DeepfakeCNN().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy (Para clasificación 0 vs 1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    epochs = 15
    print("\nIniciando entrenamiento de la CNN (PyTorch)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    # =========================================================
    # EXPORTACIÓN A ONNX (ESTÁNDAR UNIVERSAL)
    # =========================================================
    print(f"\nExportando modelo Edge AI (Formato ONNX) a {output_model_dir} ...")
    os.makedirs(output_model_dir, exist_ok=True)
    onnx_path = os.path.join(output_model_dir, 'model.onnx')
    
    # Guardado de seguridad nativo de PyTorch (por si falla ONNX)
    torch.save(model.state_dict(), os.path.join(output_model_dir, 'model_backup.pth'))

    # Creamos un tensor de muestra falso (1 imagen, 1 canal, 128x128)
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11,          # Versión altamente compatible
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Parche: Forzar a ONNX a incrustar los pesos matemáticos dentro del propio archivo
    # para que la extensión de Chrome no tenga que leer archivos externos fragmentados.
    import onnx
    print("Incrustando pesos internamente para compatibilidad web...")
    merged_model = onnx.load(onnx_path)
    onnx.save_model(merged_model, onnx_path)
    
    # Borramos la basura fragmentada si PyTorch llegó a generarla
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        
    print(f"¡Exportación Completada! Archivo unificado guardado en: {onnx_path}")
    print("Este archivo .onnx es ligero y no fallará en tu extensión.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar CNN para detectar Deepfakes con PyTorch")
    parser.add_argument('--dataset', type=str, default='../data/processed_dataset', help="Carpeta con los archivos .npy")
    parser.add_argument('--output', type=str, default='../extension/onnx_model', help="Carpeta de salida para el modelo Web")
    
    args = parser.parse_args()
    train_and_export(args.dataset, args.output)
