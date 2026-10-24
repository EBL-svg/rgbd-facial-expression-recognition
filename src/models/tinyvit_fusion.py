import timm
import torch
import torch.nn as nn

class TinyViT_Gated_Fusion(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # 1. Backbones : Initialisation de deux encodeurs TinyViT identiques
        # On utilise 'tiny_vit_5m_224' (5 millions de parametres) pour l'efficacite en temps reel
        # num_classes=0 et global_pool='' permettent d'extraire les cartes de caracteristiques brutes
        self.rgb_encoder = timm.create_model('tiny_vit_5m_224', pretrained=pretrained, num_classes=0, global_pool='')
        self.depth_encoder = timm.create_model('tiny_vit_5m_224', pretrained=pretrained, num_classes=0, global_pool='')
        
        feature_dim = 320 # Dimension des vecteurs (canaux) en sortie du bloc final de TinyViT-5M

        # 2. Module de Gating : Utilise la profondeur pour filtrer les zones importantes du RGB
        self.gate = nn.Sequential(
            # Convolution 1x1 pour transformer les caracteristiques de profondeur en masque d'attention
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            # Sigmoid ecrase les valeurs entre 0 et 1 (0 = ignore, 1 = conserve)
            nn.Sigmoid()
        )

        # 3. Fusion Transformer (Le Bottleneck)
        # Couche d'encodeur Transformer pour modeliser les relations globales entre les patchs d'image
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Jeton de fusion (similaire au [CLS] token) : parametre appris qui va condenser l'information finale
        self.fusion_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # normaliser pour stabiliser
        self.norm = nn.LayerNorm(feature_dim)

        # 4. Head : Couche lineaire finale pour la classification en 7 emotions
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, rgb, depth):
        # Pour TinyViT, la profondeur (1 canal) doit etre repetee 3 fois pour imiter un format RGB
        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)

        # Extraction des caracteristiques (features maps) : sortie de forme (B, 320, 7, 7)
        f_rgb = self.rgb_encoder(rgb)      
        f_depth = self.depth_encoder(depth) 

        # Etape de Gating : on cree un masque a partir de la profondeur et on l'applique au RGB
        mask = self.gate(f_depth)
        gated_rgb = f_rgb * mask # Multiplication element par element (Hadamard product)

        # Etape de Fusion Transformer
        # Aplatit la grille 7x7 en une sequence de 49 tokens de dimension 320
        tokens_rgb = gated_rgb.flatten(2).permute(0, 2, 1) # Resultat : (B, 49, 320)

        # de meme pour tokens_depth aussi pour la fusion
        tokens_depth = f_depth.flatten(2).permute(0, 2, 1)

        b = rgb.size(0) # Recupere la taille du batch
        # Etend le fusion_token pour qu'il corresponde a la taille du batch actuel
        f_token = self.fusion_token.expand(b, -1, -1)
        
        # Concatene le fusion_token (1) avec les tokens RGB (49) -> sequence de 50 tokens

        #tokens = torch.cat((f_token, tokens_rgb), dim=1) # (B, 50, 320)

        # on donne plutot RGB + Depth
        tokens = torch.cat((f_token, tokens_rgb, tokens_depth), dim=1)  # (B, 1+49+49, 320)
        
        # Passage dans le Transformer pour que les tokens interagissent entre eux (Self-Attention)
        fused_tokens = self.fusion_transformer(tokens)

        
        # On ne garde que la sortie correspondant au premier token (le fusion_token) pour predire l'emotion
        x = self.norm(fused_tokens[:, 0, :])
        return self.head(x)