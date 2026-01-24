"""
Late Fusion modeli => Her görüntü (LCC-RCC-RMLO-LMLO) ayrı CNN'den geçer, sonra birleştirir (Feature Concat)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path

# Local weightlerin çekildiği klasör
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "pretrained_weights"

class LateFusionBiRADS(nn.Module):
    """
    Late Fusion Model - BI-RADS Sınıflandırma
    
    Yapı:
        CC  → CNN (encoder) → Features (1280)
                                    ↓
                                  Concat (2560)
                                    ↓
        MLO → CNN (encoder) → Features (1280)
                                    ↓
                               Fusion Layer
                                    ↓
                            BI-RADS Prediction (4)
                            
    Bu benim ilk toplatıda yaptığım sunumun mimarisiydi.
    Nurullah hoca daha sonra attention mechanism ve late fusion olan bir durum söyledi, o model başka bir dosyada olacak.
    """
    
    def __init__(self, config):
        # config.py dosyasındaki Config classından inherit alırız
        super().__init__()
        
        self.config = config
        self.backbone = config.MODEL_NAME # "efficientnet_b0"
        self.num_classes = config.NUM_CLASSES # 4 (BI-RADS 1-2-4-5)
        
        # CC VE MLO için ayrı encoder
        self.cc_encoder = self._create_encoder()
        self.mlo_encoder = self._create_encoder()
        
        # Feature Dimension
        feature_dim = self._get_feature_dim(self.backbone) # Mesela efficientnet b0 için 1280
        
        # Fusion Classifier
        dropout = 0.3
        self.fusion = nn.Sequential(
          nn.Linear(feature_dim * 2, 512), # feature_dim x 2 = 1280 x 2 = 2560 ------- 2560 -> 512
          nn.BatchNorm1d(512),
          nn.ReLU(inplace = True),
          nn.Dropout(dropout),
          
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(inplace = True),
          nn.Dropout(dropout),
          
          nn.Linear(256, self.num_classes) # 256 -> 4  
        )
        
        print(f"Late Fusion Model:")
        print(f" Backbone: {self.backbone}")
        print(f" Feature Dim: {feature_dim}")
        print(f" Fusion Input: {feature_dim * 2}")
        print(f" Num classes: {self.num_classes}")
        
    def _create_encoder(self):
        """Feature exteaction için encoder oluşturur"""

        if self.config.PRETRAINED:
            # Localden wieghtleri çek
            encoder = timm.create_model(
                self.backbone,
                pretrained = False,
                num_classes = 0, # Feature extractor (classifier katmanı kaldırılır, çünkü modelin son layerını kullanmıyoruz, sadece feature'ları çıkarıp kendi classifierımızla (fusion) kullanıyoruz)
                in_chans = 3 # Önce 3 kanal yapıyoruz çünkü imagenet böyle train edilmiş
            )

            weight_path = WEIGHTS_DIR / f"{self.backbone}.pth"
            if weight_path.exists():
                state_dict = torch.load(weight_path, map_location = 'cpu')
                encoder.load_state_dict(state_dict, strict = False)

            # First conv layer'ı channel'a adapte eder
            # Elimizdeki görüntüler 1 kanallı, ama pretrained model 3 kanallı
            # First conv layerı 3 kanaldan 1 kanala adapte ediyoruz
            encoder = self._adapt_first_conv(encoder, self.config.IN_CHANNELS)

        else:
            # Random initialization uygular
            encoder = timm.create_model(
                self.backbone,
                pretrained = False,
                num_classes = 0,
                in_chans = self.config.IN_CHANNELS
            )

        return encoder
    
    def _adapt_first_conv(self, model, in_channels):
        """First conv layer'ı görüntüye (kanala) adapte eder"""

        # EfficientNet
        if hasattr(model, 'conv_stem'):
            old_conv = model.conv_stem # First conv layer (3 channels)
            new_conv = nn.Conv2d(
                in_channels, # 1 channel
                old_conv.out_channels, # 32 (efficientnet b0)
                kernel_size = old_conv.kernel_size,
                stride = old_conv.stride,
                padding = old_conv.padding,
                bias = False
            )
            with torch.no_grad():
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            model.conv_stem = new_conv

        # ResNet
        elif hasattr(model, 'conv1'):
            old_conv = model.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size = old_conv.kernel_size,
                stride = old_conv.stride,
                padding = old_conv.padding,
                bias = False
            )
            with torch.no_grad():
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            model.conv1 = new_conv

        # ConvNeXt
        elif hasattr(model, 'stem') and hasattr(model.stem, '0'):
            old_conv = model.stem[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size = old_conv.kernel_size,
                stride = old_conv.stride,
                padding = old_conv.padding,
                bias = False
            )
            # old_conv.weight = (32, 3, 3, 3) -> 32 output channel, 3 input channel, 3x3 kernel size
            # new_conv.weight = (32, 1, 3, 3) -> 1 input channel
            with torch.no_grad():
              # ilk in_channels kanalını kopyalar
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            model.stem[0] = new_conv # RGB'ni (3 channels) first layerını Grayscale'e (1 channel) kopyalar, bu sayede pretrained bilgi kaybolmaz.

        # Swin Transformer
        elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
            old_conv = model.patch_embed.proj
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size = old_conv.kernel_size,
                stride = old_conv.stride,
                padding = old_conv.padding,
                bias = old_conv.bias is not None
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
        """Backbone'a göre feature dimension oluşturur"""

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

    def forward(self, cc_img, mlo_img):
        """
        Input:

          - cc_img: (4, 1, 384, 384)
              - 4: batch size
              - 1: grayscale
              - 384x384: görüntü boyutu
          mlo_img: (4, 1, 384, 384)
          
          1. Feature Extraction:

          cc_img  → cc_encoder  → cc_features  (4, 1280)
          mlo_img → mlo_encoder → mlo_features (4, 1280)
          Her CNN, görüntüyü 1280 boyutlu bir vektöre sıkıştırır.

          2. Concatenation:
          
          combined = torch.cat([cc_features, mlo_features], dim=1)
          # (4, 1280) + (4, 1280) = (4, 2560)
          İki feature vektörü yan yana birleştirilir.

          3. Classification:

          combined (4, 2560) → fusion → output (4, 4)
          Her örnek için 4 BI-RADS sınıfı skorları.
        """
        #Feature extration
        cc_features = self.cc_encoder(cc_img) # (batch, 1280)
        mlo_features = self.mlo_encoder(mlo_img) # (batch, 1280)

        # Feature concat
        combined = torch.cat([cc_features, mlo_features], dim = 1) # (batch, 2560)

        # Classification
        output = self.fusion(combined) # (batch, 4)

        return output

def get_late_fusion_model(config):
    """
    Late Fusion modeli döndürür
    config.py içerisindeki parametreleri alır
    """
    return LateFusionBiRADS(config)

# Test kodu
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import Config
    
    print("\n" + "="*60)
    print("LATE FUSION MODEL TEST")
    print("="*60 + "\n")
    
    # Model oluştur
    Config.PRETRAINED = True
    model = get_late_fusion_model(Config)
    
    # Dummy input
    cc_batch = torch.randn(4, 1, 384, 384)
    mlo_batch = torch.randn(4, 1, 384, 384)

    print(f"\nInput shapes:")
    print(f"   CC: {cc_batch.shape}")
    print(f"   MLO: {mlo_batch.shape}")

    # Forward pass
    output = model(cc_batch, mlo_batch)

    print(f"\nOutput shape: {output.shape}")
    print(f"   Beklenen: (4, 4)")

    assert output.shape == (4, 4), "Output shape yanlis!"

    print("\nLate fusion model testi basarili!")