import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class BinaryImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # Forzar explícitamente solo estas dos carpetas como clases
        classes = ['fake', 'reales']
        class_to_idx = {'fake': 0, 'reales': 1}
        return classes, class_to_idx

def get_dataloaders_and_loss(data_dir, batch_size=32, img_size=(224, 224), val_split=0.2):
    """
    Carga el dataset desde la estructura de carpetas, crea dataloaders para entrenamiento y
    validación, y calcula los pesos de clase para compensar el desbalanceo.
    """
    # Transformaciones básicas (ajustar según el modelo, ej: ResNet, EfficientNet, etc.)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet default
                             std=[0.229, 0.224, 0.225])
    ])

    # Cargar todo el dataset usando BinaryImageFolder
    # Solo mirará las carpetas 'fake' (0) y 'reales' (1)
    print(f"Cargando dataset desde: {data_dir}...")
    full_dataset = BinaryImageFolder(root=data_dir, transform=transform)
    
    class_names = full_dataset.classes
    print(f"Clases detectadas: {class_names}")
    print(f"Total de imágenes: {len(full_dataset)}")

    # Dividir en entrenamiento y validación
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Crear Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- CÁLCULO DE PESOS DE CLASE ---
    # Extraer las etiquetas de la partición de entrenamiento
    print("Calculando pesos de clase para compensar el desbalanceo...")
    
    # Extraemos solo los índices que pertenecen al train_dataset
    train_indices = train_dataset.indices
    train_labels = [full_dataset.targets[i] for i in train_indices]
    
    # Calcular pesos usando scikit-learn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    # Convertir a tensor de PyTorch (y mover a GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Pesos calculados: {class_weights}")
    
    # Definir la función de pérdida con los pesos
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    return train_loader, val_loader, criterion, device

if __name__ == '__main__':
    # Script de prueba rápida
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    print("=== Probando DataLoader y Función de Pérdida ===")
    train_loader, val_loader, criterion, device = get_dataloaders_and_loss(
        data_dir=DATA_DIR, 
        batch_size=32
    )
    print("Todo listo para entrenar el modelo.")
