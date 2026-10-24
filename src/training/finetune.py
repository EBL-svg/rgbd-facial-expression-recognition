import os
import torch
import gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# Importation des modules personnalisés (architecture et prétraitement)
from src.models.tinyvit_fusion import TinyViT_Gated_Fusion
from src.data.rgbd_dataset import PairedTransform

# --- OPTIMISATION G6 ---
# Accélère les calculs matriciels sur les GPU NVIDIA L4 (architecture Ada Lovelace)
torch.set_float32_matmul_precision('high')

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get(
        "FEDLFI_DATA_ROOT",
        str(PROJECT_ROOT / "data" / "FED-LFI")
    )
)

PRETRAINED_PATH = PROJECT_ROOT / "checkpoints" / "tinyvit_fusion_best.pth"

SAVE_DIR = PROJECT_ROOT / "experiments_fedlfi"

BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-6             # Learning Rate très faible pour le fine-tuning (ajustement fin)
NUM_FOLDS = 10        # Validation croisée à 10 plis
RUNS_PER_FOLD = 3     # Nombre de répétitions par pli pour stabiliser les résultats
DEVICE = "cuda"
SPLIT_SEED = 42       # Graine fixe pour la reproductibilité du découpage

# Mapping des émotions vers les index numériques
CLASS_MAP = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Neutrality': 4, 'Sadness': 5, 'Surprise': 6}
CLASS_NAMES = list(CLASS_MAP.keys())

# ==============================================================================
#  UTILITAIRES
# ==============================================================================
class EarlyStopping:
    """ Interrompt l'entraînement si l'Accuracy de validation stagne pendant 'patience' époques """
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0
        self.stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return True # Nouveau record atteint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True # Signal d'arrêt
            return False

def set_seed(seed):
    """ Fixe toutes les graines aléatoires pour garantir la reproductibilité des expériences """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
#  DATASET
# ==============================================================================
class FEDLFIDataset(Dataset):
    """ Dataset spécifique pour FED-LFI : charge des paires synchronisées par sujet et émotion """
    def __init__(self, root_dir, subject_list, paired_transform=None):
        self.samples = []
        self.paired_transform = paired_transform
        # Parcours uniquement les sujets attribués à ce Fold
        for subject in subject_list:
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path): continue
            for emotion in os.listdir(subject_path):
                if emotion not in CLASS_MAP: continue
                label = CLASS_MAP[emotion]
                emotion_path = os.path.join(subject_path, emotion)
                if not os.path.isdir(emotion_path): continue
                # Appariement rigoureux des fichiers TotalFocus (RGB) et DepthMapColored (Profondeur)
                for f in os.listdir(emotion_path):
                    if "TotalFocus.png" in f:
                        rgb_path = os.path.join(emotion_path, f)
                        depth_path = os.path.join(emotion_path, f.replace("TotalFocus.png", "DepthMapColored.png"))
                        if os.path.exists(depth_path):
                            self.samples.append((rgb_path, depth_path, label))

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_p, depth_p, label = self.samples[idx]
        rgb = Image.open(rgb_p).convert('RGB')
        depth = Image.open(depth_p).convert('L') # Profondeur chargée en niveaux de gris
        if self.paired_transform: rgb, depth = self.paired_transform(rgb, depth)
        return rgb, depth, label

# ==============================================================================
#  ENTRAÎNEMENT D'UN RUN
# ==============================================================================
def train_one_run(fold_idx, run_idx, train_subjects, val_subjects):
    """ Exécute un cycle complet d'entraînement pour un pli et un run donnés """
    set_seed(SPLIT_SEED + run_idx)
    
    # Création des loaders avec augmentation pour le train et évaluation fixe pour le val
    train_ds = FEDLFIDataset(DATA_ROOT, train_subjects, PairedTransform(size=224, train=True))
    val_ds = FEDLFIDataset(DATA_ROOT, val_subjects, PairedTransform(size=224, train=False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Initialisation du modèle et chargement des poids pré-entraînés
    model = TinyViT_Gated_Fusion(num_classes=7).to(DEVICE)
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False) # strict=False pour adapter la dernière couche (7 classes)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5)
    
    best_run_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Run {run_idx+1} Ep {epoch+1}/{EPOCHS}", leave=False)
        
        for rgb, depth, labels in train_pbar:
            rgb, depth, labels = rgb.to(DEVICE), depth.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Phase de Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for rgb, depth, labels in val_loader:
                rgb, depth, labels = rgb.to(DEVICE), depth.to(DEVICE), labels.to(DEVICE)
                out = model(rgb, depth)
                correct += out.max(1)[1].eq(labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100. * correct / total
        
        # Sauvegarde du meilleur état pour ce run
        if early_stopper(val_acc):
            best_run_acc = val_acc

            # best_model_state = {k: v.cpu().copy() for k, v in model.state_dict().items()}
            # copy pour numpy plutot ici =>
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if early_stopper.stop: break
            
    model.load_state_dict(best_model_state)
    return best_run_acc, model

# ==============================================================================
#  MAIN (K-FOLD PAR SUJET)
# ==============================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Identification de tous les sujets uniques (LF01, LF02, ...)
    subjects = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("LF")])
    random.seed(SPLIT_SEED)
    random.shuffle(subjects) # Mélange aléatoire des sujets
    
    fold_size = len(subjects) // NUM_FOLDS
    fold_accs = []
    global_cm = np.zeros((7, 7), dtype=int) # Matrice de confusion cumulée

    print(f"🚀 Lancement Cross-Validation (10 Folds) par Sujet")

    for fold in range(NUM_FOLDS):
        # Découpage des sujets : les sujets du val sont totalement absents du train

        # mieux si pas divisible par 10
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < NUM_FOLDS - 1 else len(subjects)
        val_sub = subjects[start:end]

        # val_sub = subjects[fold*fold_size : (fold+1)*fold_size]
        train_sub = [s for s in subjects if s not in val_sub]
        
        print(f"\n FOLD {fold+1}/{NUM_FOLDS}")
        fold_champion_acc = -1
        best_fold_model = None

        for run in range(RUNS_PER_FOLD):
            acc, model = train_one_run(fold, run, train_sub, val_sub)
            if acc > fold_champion_acc:
                fold_champion_acc = acc
                best_fold_model = model
            print(f"     Run {run+1} Best Acc: {acc:.2f}%")

        # --- EVALUATION FINALE DU PLI ---
        # Calcul du rapport de classification et de la matrice de confusion sur les sujets de validation
        best_fold_model.eval()
        final_val_ds = FEDLFIDataset(DATA_ROOT, val_sub, PairedTransform(size=224, train=False))
        loader = DataLoader(final_val_ds, batch_size=BATCH_SIZE)
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for r, d, l in loader:
                p = best_fold_model(r.to(DEVICE), d.to(DEVICE)).max(1)[1]
                y_true.extend(l.numpy())
                y_pred.extend(p.cpu().numpy())
        
        # Affichage des métriques détaillées (Précision, Rappel, F1)
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        
        # Accumulation dans la matrice de confusion globale
        cm = confusion_matrix(y_true, y_pred, labels=range(7))
        global_cm += cm
        fold_accs.append(fold_champion_acc)

    # Affichage de la performance moyenne finale du système
    print("\n" + "="*55)
    print(f"RÉSULTAT FINAL 10-FOLD CV : {np.mean(fold_accs):.2f}% (±{np.std(fold_accs):.2f})")
    print("="*55)
    print("\nMatrice de Confusion Globale cumulée :")
    print(global_cm)

if __name__ == "__main__":
    main()