import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resize_with_padding(
    img,
    target_size,
    fill=0,
    interpolation=InterpolationMode.BILINEAR
):
    """
    Resize en conservant le ratio + padding pour obtenir un carré target_size x target_size.
    fill=0 -> padding noir.
    """
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Image with zero width/height")

    scale = target_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize avec le bon paramètre nommé: interpolation=
    img = transforms.functional.resize(
        img,
        [new_h, new_w],
        interpolation=interpolation
        )

    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top

    img = transforms.functional.pad(
        img,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=fill
    )
    return img


class PairedTransform:
    """
    Classe pour appliquer des transformations identiques au couple RGB et Profondeur.
    Garantit l'alignement spatial des pixels entre les deux sources.
    """
    def __init__(self, size=224, train=True):
        self.size = size  # Taille finale de l'image (ex: 224x224)
        self.train = train # Définit si on est en mode entrainement (augmentation) ou validation

    def __call__(self, rgb, depth):
        # 1. Resize de base : on agrandit un peu avant de cropper pour garder de la marge
        scale_size = int(self.size * 1.14)
        rgb = resize_with_padding(
            rgb,
            scale_size,
            fill=0,
            interpolation=InterpolationMode.BILINEAR
        )  # padding noir, interpolation bilinear

        depth = resize_with_padding(
            depth,
            scale_size,
            fill=0,
            interpolation=InterpolationMode.NEAREST
        )  # padding noir, interpolation NEAREST

        if self.train:
            # 2. Random Crop synchronisé : on calcule les paramètres une seule fois
            i, j, h, w = transforms.RandomCrop.get_params(
                rgb,
                output_size=(self.size, self.size)
            )
            rgb = transforms.functional.crop(rgb, i, j, h, w)   # Coupe le RGB selon (i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w) # Coupe la Depth avec les memes coordonnees

            # 3. Random Horizontal Flip synchronisé : pile ou face (0.5)
            if torch.rand(1).item() < 0.5:
                rgb = transforms.functional.hflip(rgb)   # Retourne le RGB horizontalement
                depth = transforms.functional.hflip(depth) # Retourne la Depth horizontalement
        else:
            # Center Crop pour la validation : on prend le milieu de l'image pour un test stable
            rgb = transforms.functional.center_crop(rgb, (self.size, self.size))
            depth = transforms.functional.center_crop(depth, (self.size, self.size))

        # 4. Conversion en Tensor : transforme les pixels (0-255) en valeurs (0.0-1.0)
        rgb = transforms.functional.to_tensor(rgb)
        depth = transforms.functional.to_tensor(depth)

        # 5. Normalisation ImageNet (RGB uniquement) : standardise les couleurs selon les stats ImageNet
        rgb = transforms.functional.normalize(
            rgb,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return rgb, depth  # Retourne le couple prêt pour le modèle


class DualImageFolder(Dataset):
    """
    Dataset personnalisé pour charger des paires d'images stockées dans deux dossiers racines différents.
    """
    def __init__(self, rgb_root, depth_root, paired_transform=None):
        self.rgb_root = rgb_root     # Chemin vers les dossiers RGB
        self.depth_root = depth_root   # Chemin vers les dossiers de profondeur
        self.paired_transform = paired_transform # La transformation synchrone definie plus haut
        
        # Liste les noms des classes (noms des dossiers)
        self.classes = sorted([d for d in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, d))])
        # Cree un dictionnaire associant chaque nom de classe a un index (ex: 'Anger' -> 0)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = [] # Liste qui contiendra les triplets (chemin_rgb, chemin_depth, label)
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG') # Extensions autorisees
        
        print(f"Scan des dossiers en cours...")
        for cls_name in self.classes:
            rgb_class_dir = os.path.join(rgb_root, cls_name)     # Chemin complet du dossier RGB de la classe
            depth_class_dir = os.path.join(depth_root, cls_name) # Chemin complet du dossier Depth de la classe
            
            if not os.path.exists(depth_class_dir): continue # Si le dossier de profondeur manque, on ignore

            # On indexe les fichiers depth dans un 'set' pour une recherche tres rapide (O(1))
            depth_files = set(os.listdir(depth_class_dir))

            for fname in os.listdir(rgb_class_dir):
                if fname.lower().endswith(valid_ext): # Verifie si c'est bien une image
                    base_name = os.path.splitext(fname)[0] # Recupere le nom de fichier sans extension
                    
                    # On cherche la profondeur correspondante en testant plusieurs extensions possibles
                    found = False
                    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
                        target_depth = base_name + ext # Reconstruit le nom de fichier depth cible
                        if target_depth in depth_files: # Verifie la presence dans le set
                            rgb_path = os.path.join(rgb_class_dir, fname)      # Chemin final RGB
                            depth_path = os.path.join(depth_class_dir, target_depth) # Chemin final Depth
                            # Ajoute le couple valide a notre liste d'echantillons
                            self.samples.append((rgb_path, depth_path, self.class_to_idx[cls_name]))
                            found = True
                            break
        print(f"Total paires trouvées : {len(self.samples)}")

    def __len__(self):
        return len(self.samples) # Retourne la taille totale du dataset

    def __getitem__(self, idx):
        # Recupere les chemins et le label pour l'index donne
        rgb_path, depth_path, label = self.samples[idx]
        
        # Ouvre les images avec PIL (Python Imaging Library)
        rgb = Image.open(rgb_path).convert('RGB') # Charge en couleur (3 canaux)
        depth = Image.open(depth_path).convert('L') # Charge en niveaux de gris (1 seul canal de profondeur)

        # Applique les transformations (Resize, Crop, Normalisation) si definies
        if self.paired_transform:
            rgb, depth = self.paired_transform(rgb, depth)
            
        return rgb, depth, label # Retourne le batch au DataLoader