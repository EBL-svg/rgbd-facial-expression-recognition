import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import de tes modules personnalises
from src.models.tinyvit_fusion import TinyViT_Gated_Fusion
from src.data.rgbd_dataset import DualImageFolder, PairedTransform

# Training was accelerated using NVIDIA Tensor Cores by enabling TF32 and mixed-precision 
#computation (AMP), yielding significant throughput improvements without measurable accuracy degradation.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION AWS ---
BATCH_SIZE = 32          # Nombre d'images traitees simultanement par le GPU
ACCUMULATION_STEPS = 2   # Simule un batch plus grand (64) en accumulant les gradients sur 2 etapes
NUM_WORKERS = 8          # Nombre de coeurs CPU utilises pour charger les donnees
EPOCHS = 50              # Nombre maximum de passages sur le dataset complet
LR = 1e-4                # Taux d'apprentissage de base (Learning Rate)
WEIGHT_DECAY = 0.05      # Regularisation L2 pour eviter le sur-apprentissage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Utilise le GPU NVIDIA L4 si disponible

# Chemins des donnees sur l'instance AWS

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset root can be configured through an environment variable.
DATA_ROOT = Path(
    os.environ.get("PFE_DATA_ROOT", str(PROJECT_ROOT / "data"))
)

TRAIN_RGB_DIR = DATA_ROOT / "AffectNet" / "AffectNet"
TRAIN_DEPTH_DIR = DATA_ROOT / "AffectNet_Depth" / "AffectNet_da_vitb"

VAL_RGB_DIR = DATA_ROOT / "AffectNet" / "AffectNet_valid"
VAL_DEPTH_DIR = DATA_ROOT / "AffectNet_Depth" / "AffectNet_valid_da_vitb"

SAVE_DIR = PROJECT_ROOT / "checkpoints"
# ==============================================================================
#  UTILITAIRES
# ==============================================================================
class EarlyStopping:
    """ Arrete l'entrainement si la precision (Acc) ne progresse plus après 'patience' epoques """
    def __init__(self, patience=8, min_delta=0.01):
        self.patience = patience    # Nombre d'epoques a attendre sans progres
        self.min_delta = min_delta  # Seuil minimal d'amelioration
        self.counter = 0            # Compteur d'epoques stagnantes
        self.best_acc = None        # Meilleur score atteint
        self.early_stop = False     # Drapeau d'arret

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True # Declenchement de l'arret
        else:
            self.best_acc = val_acc
            self.counter = 0 # Reinitialisation si progres reel

def save_plots(history, path="training_curves.png"):
    """ Genere et sauvegarde les courbes de Loss et d'Accuracy """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss History')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig(path)

# ==============================================================================
#  BOUCLE PRINCIPALE
# ==============================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Datasets & Dataloaders (Chargement multimodal synchronise)
    train_transform = PairedTransform(size=224, train=True)
    val_transform = PairedTransform(size=224, train=False)

    train_ds = DualImageFolder(TRAIN_RGB_DIR, TRAIN_DEPTH_DIR, paired_transform=train_transform)
    val_ds = DualImageFolder(VAL_RGB_DIR, VAL_DEPTH_DIR, paired_transform=val_transform)

    # Optimisation des flux : pin_memory=True accelere le transfert CPU -> GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Modele & Optimisation differentielle
    model = TinyViT_Gated_Fusion(num_classes=len(train_ds.classes), pretrained=True).to(DEVICE)
    
    # On applique un LR plus faible sur les backbones deja pre-entraines (ImageNet)
    # et un LR standard sur les nouveaux modules de fusion (Gating, Transformer)
    optimizer = optim.AdamW([
        {'params': model.rgb_encoder.parameters(), 'lr': LR * 0.1},
        {'params': model.depth_encoder.parameters(), 'lr': LR * 0.1},
        {'params': model.gate.parameters(), 'lr': LR},
        {'params': model.fusion_transformer.parameters(), 'lr': LR},
        {'params': [model.fusion_token], 'lr': LR},
        {'params': model.head.parameters(), 'lr': LR}
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss() # Fonction de perte pour la classification
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) # Decroissance douce du LR
    scaler = torch.amp.GradScaler("cuda") # Utilise pour la precision mixte (Mixed Precision)
    early_stopper = EarlyStopping(patience=10)

    # 3. Suivi des performances
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # 4. Entraînement
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (rgb, depth, labels) in enumerate(loop):
            rgb, depth, labels = rgb.to(DEVICE), depth.to(DEVICE), labels.to(DEVICE)

            # Autocast : calculs en Float16 pour gagner en vitesse et en memoire GPU
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                outputs = model(rgb, depth)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            # Retropropagation du gradient echelle (scaled)
            scaler.scale(loss).backward()

            # Mise a jour des poids toutes les 'ACCUMULATION_STEPS' etapes
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Reinitialisation des gradients

            # Stats par iteration
            running_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            current_acc = 100. * correct / total
            loop.set_postfix(loss=f"{running_loss/(i+1):.4f}", acc=f"{current_acc:.2f}%")

        # 5. Validation de fin d'époque
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for rgb, depth, labels in val_loader:
                rgb, depth, labels = rgb.to(DEVICE), depth.to(DEVICE), labels.to(DEVICE)
                outputs = model(rgb, depth)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
        f"Best Val Acc: {best_val_acc:.2f}%")

        
        # Sauvegarde des statistiques et du meilleur modele
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "tinyvit_fusion_best.pth")
            torch.save({'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc}, save_path)


        early_stopper(val_acc)
        if early_stopper.early_stop:
           print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
           break
        scheduler.step()

    save_plots(history)
    
    torch.save(history, os.path.join(SAVE_DIR, "training_history.pth"))

    
    
if __name__ == "__main__":
    main()
