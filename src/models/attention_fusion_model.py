"""
Attention + late-fusion model

Model attention mekanizmasını kullanarak hangi görünümün daha önemli olduğunu öğrenir

"""

import torch
import torch.nn as nn
import timm
from pathlib import  Path

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "pretrained_weights"

class AttentionFusion(nn.Module):
    """
    CC ve MLO görüntülerine farklı weightler öğrenir
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        
        # Attention weightlerini öğrenir
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), # 2 feature'ı birlştir
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(256, 2), # 2 view için (CC - MLO)
            nn.Softmax(dim = 1) # Toplam weightleri 1 yap
        )
        
    def forward(self, cc_features, mlo_features):
        """
        Input:
            - cc_features = (batch, feature_dim)
            - mlo_features = (batch, feature_dim)
        Output:
            - attended_features = (batch, feature_dim)  --- Weighted features
            - attention_weights = (batch, 2) --- [CC_weight, MLO_weight]
        """
        
        # İki feature'ı birleştirir
        concat = torch.cat([cc_features, mlo_features], dim = 1) # (batch, feature_dim * 2)
        
        # Attention weightlerini hesapla (örn: Model CC görüntüsünde daha belirgin bir lezyon gördüyse => CC = 0.7 ,MLO = 0.3)
        attention_weights = self.attention(concat) # (batch, 2)
        
        # Weighted toplam
        cc_weight = attention_weights[:, 0:1] # (batch, 1)
        mlo_weight = attention_weights[:, 1:2] # (batch, 1)
        
        attended = (cc_weight * cc_features) + (mlo_weight * mlo_features)
        
        return attended, attention_weights
    
class AttentionLateFusionBiRADS(nn.Module):
    """
    Yapı:
        CC  → Encoder → Features ─┐
                                   ├→ Attention → Weighted Features → Classifier → BI-RADS
        MLO → Encoder → Features ─┘
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.backbone = config.MODEL_NAME
        self.num_classes = config.NUM_CLASSES
        
        # CC - MLO Encoders
        self.cc_encoder = self._create_encoder()
        self.mlo_encoder = self._create_encoder()
        
        # Feature dimension
        feature_dim = self._get_feature_dim(self.backbone)
        
        # Attention fusion
        self.attention_fusion = AttentionFusion(feature_dim)

        # Classifier (artık 2x değil 1x feature dim)
        dropout = 0.3
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            
            nn.Linear(256, self.num_classes)
        ) 
        
        print(f"Attention Late Fusion Model initialized:")
        print(f"   Backbone: {self.backbone}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Attention-weighted fusion")
        print(f"   Num classes: {self.num_classes}")
        
    def _create_encoder(self):
        """Encoderları oluşturur (feature extractor)"""
        if self.config.PRETRAINED:
            # Local weightleri yükle
            encoder = timm.create_model(
                self.backbone,
                pretrained = False,
                num_classes = 0, # CNN modelinin son layer'ı kaldırılır, sadece Feature extractor olur ve feature vektörü döner
                in_chans = 3 # Önce 3 channel
            )
            
            # Local weightleri yükle 
            weight_path = WEIGHTS_DIR / f"{self.backbone}.pth"
            if weight_path.exists():
                state_dict = torch.load(weight_path, map_location = 'cpu')
                encoder.load_state_dict(state_dict, strict = False)
                
            # First conv layer'ı 1 kanala adapte et
            encoder = self._adapt_first_conv(encoder, self.config.IN_CHANNELS)
            
            
        else:    
            encoder = timm.create_model(
                self.backbone,
                pretrained = False,
                num_classes = 0,
                in_chans = self.config.IN_CHANNELS
            )
        
        return encoder
    
    def _adapt_first_conv(self, model, in_channels):
        """İlk conv layer'ı adapte et"""
        
        # EfficientNet
        if hasattr(model, 'conv_stem'):
            old_conv = model.conv_stem
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            model.conv_stem = new_conv
        
        # ResNet
        elif hasattr(model, 'conv1'):
            old_conv = model.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            model.conv1 = new_conv
        
        # ConvNeXt
        elif hasattr(model, 'stem') and hasattr(model.stem, '0'):
            old_conv = model.stem[0]
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            model.stem[0] = new_conv

        # Swin Transformer
        elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
            old_conv = model.patch_embed.proj
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            model.patch_embed.proj = new_conv

        return model
    
    def _get_feature_dim(self, backbone):
        """Backbone'e göre feature dimension"""
        if 'efficientnet_b0' in backbone:
            return 1280
        elif 'resnet50' in backbone:
            return 2048
        elif 'convnext' in backbone:
            return 768
        elif 'swin' in backbone:
            return 768
        elif 'vit' in backbone:
            return 384
        else:
            return 1280
    
    def forward(self, cc_img, mlo_img, return_attention=False):
        """
        Forward pass
        
        Input:
            cc_img: (batch, 1, H, W) - CC görüntüleri
            mlo_img: (batch, 1, H, W) - MLO görüntüleri
            return_attention: Attention ağırlıklarını döndür mü?
        
        Output:
            output: (batch, num_classes) - BI-RADS logits
            attention_weights (opsiyonel): (batch, 2) - [CC_weight, MLO_weight]
        """
        # Feature extraction
        cc_features = self.cc_encoder(cc_img)
        mlo_features = self.mlo_encoder(mlo_img)
        
        # Attention fusion
        attended_features, attention_weights = self.attention_fusion(
            cc_features, mlo_features
        )
        
        # Classification
        output = self.classifier(attended_features)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def get_attention_weights(self, cc_img, mlo_img):
        """
        Sadece attention ağırlıklarını döndür (analiz için)
        
        Input:
            cc_img: (batch, 1, H, W)
            mlo_img: (batch, 1, H, W)
        
        Output:
            attention_weights: (batch, 2)
                [:, 0] → CC view importance
                [:, 1] → MLO view importance
        """
        with torch.no_grad():
            _, attention_weights = self.forward(
                cc_img, mlo_img, return_attention=True
            )
        return attention_weights


def get_attention_fusion_model(config):
    """
        AttentionLateFusionBiRADS modelini döndürür
    """
    return AttentionLateFusionBiRADS(config)


# Test kodu
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import Config
    
    print("\n" + "="*60)
    print("ATTENTION FUSION MODEL TEST")
    print("="*60 + "\n")
    
    # Model oluştur
    Config.PRETRAINED = True
    model = get_attention_fusion_model(Config)
    
    # Dummy input
    cc_batch = torch.randn(4, 1, 384, 384)
    mlo_batch = torch.randn(4, 1, 384, 384)
    
    print(f"\n Input shapes:")
    print(f"   CC: {cc_batch.shape}")
    print(f"   MLO: {mlo_batch.shape}")
    
    # Forward pass (normal)
    output = model(cc_batch, mlo_batch)
    print(f"\n Output shape: {output.shape}")
    print(f"   Beklenen: (4, 4)")
    
    # Forward pass (with attention)
    output, attention = model(cc_batch, mlo_batch, return_attention=True)
    print(f"\n Attention weights shape: {attention.shape}")
    print(f"   Beklenen: (4, 2)")
    
    print(f"\n   Örnek attention weights (ilk 2 örnek):")
    for i in range(min(2, attention.shape[0])):
        cc_weight = attention[i, 0].item()
        mlo_weight = attention[i, 1].item()
        print(f"   Örnek {i+1}: CC={cc_weight:.3f}, MLO={mlo_weight:.3f}")
    
    # Toplam kontrolü (softmax olduğu için toplamı 1 olmalı)
    weights_sum = attention.sum(dim=1)
    print(f"\n   Attention weights toplamı: {weights_sum[0]:.3f} (1.0 olmalı)")
    
    assert output.shape == (4, 4), " Output shape yanlış!"
    assert attention.shape == (4, 2), " Attention shape yanlış!"
    assert torch.allclose(weights_sum, torch.ones(4), atol=1e-6), " Attention weights toplamı 1 değil!"
    
    print("\n Attention fusion model testi başarılı!")        

        