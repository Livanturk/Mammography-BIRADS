"""
Early Fusion
Config-aware model oluşturma
"""

import timm
import torch
import torch.nn as nn
from pathlib import Path

# İnternetten dinamik olarak çekemediğmiz için local olarak transfer learning weightlerini tutmamız gerekiyor.
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "pretrained_weights"

def load_pretrained_weights(model, model_name):
    """Pretrained weightleri klasörden çeker"""
    weight_path = WEIGHTS_DIR / f"{model_name}.pth"
    
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights bulunamadı: {weight_path}\n"
            f"Çözüm: python scripts/download_weights/requests.py çalıştır."
        )
        
    print(f"Local weightler yükleniyor: {weight_path.name}")
    
    # State dict yükleme
    state_dict = torch.load(weight_path, map_location = 'cpu')
    
    # Model'e yükle (strict = False diyerek eksik veya fazla key sorununu çözebiliriz)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)
    
    if missing_keys:
        print(f"Eksik key: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
    print(f"Weightler yüklendi")
    
    return model

def adapt_classifier(model, num_classes):
    """
    Classifier'ı yeni num_classes'a adapte eder
    """
    
    # EfficientNet
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        print(f"Classifier adapte edildi: {in_features} -> {num_classes}")
        
    # ResNet
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        print(f"Classifier adapte edildi: {in_features} -> {num_classes}")
        
    # ConvNeXt, Swin, ViT
    elif hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        print(f"Classifier adapte edildi: {in_features} -> {num_classes}")
    
    return model

def adapt_first_conv(model, in_channels):
    """
    İlk layer'ı in_channels'a göre adapte eder
    Elimizdeki PNG 1 kanallı (early fusion varsa 2) ve pretrained modeller 3 kanallı
    Biz de pretrained modelin ilk layerını düşürürüz 
    """
    
    # EfficientNet
    if hasattr(model, 'conv_stem'):
        old_conv = model.conv_stem
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride = old_conv.stride,
            padding = old_conv.padding,
            bias = False
        )
        
        # Weightleri adapte et
        with torch.no_grad():
            if in_channels <= 3:
                # Kanal azaltma (İlk in_channels kadarını kopyalar)
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            else:
                # Kanal arttırma: Loop ile tekrarlanır
                for i in range(in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]
        
        model.conv_stem = new_conv
    
    #ResNet
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
            if in_channels <=3:
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            else:
                for i in range(in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]
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
        
        with torch.no_grad():
            if in_channels <= 3:
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            else:
                for i in range(in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]

        model.stem[0] = new_conv
        
        print(f"First conv: 3 -> {in_channels} kanal")
    return model

def get_model(config):
    """
    Early fusion için model factory
    
    Config'den parametreleri alır ve local weightleri kullanır.
    
    örn: from config import Config
         model = get_model(Config)
    """
    
    # Bilateral = 2 channel (CC + MLO) / Multi-view 4 kanal (RCC-LCC-RMLO-LMLO)
    in_channels = 2 if config.APPROACH == "bilateral" else 4
    
    print(f"Model ismi: {config.MODEL_NAME}")
    print(f"Pretrained: {config.PRETRAINED} (local)")
    print(f"In Channels: {in_channels}")
    print(f"Num classes: {config.NUM_CLASSES}")
    
    try:
        # localden pretrained weightleri kullanır
        if config.PRETRAINED:
            # Önce boş bir model oluşturur (PRETRAINED = False)
            model = timm.create_model(
                config.MODEL_NAME,
                pretrained = False,
                num_classes = 1000, # ImageNet (weights için gerekli)
                in_chans = 3 # Pretrained için in_chans = 3 olmalı            
            )
            
            # Local weightleri yükler
            model = load_pretrained_weights(model, config.MODEL_NAME)
            
            # Classifier değiştir (1000 -> 4 (BI-RADS))
            model = adapt_classifier(model, config.NUM_CLASSES)
            
            # First convolutional layer'ı adapte et (3 -> 1 veya 3 -> 2)
            if in_channels != 3:
                print(f"First conv layer {in_channels} kanala adapte ediliyor.")
                model = adapt_first_conv(model, in_channels)
        # Pretrained yoksa (random initialization uyguluyor)   
        else:
            model = timm.create_model(
                config.MODEL_NAME,
                pretrained = False,
                num_classes = config.NUM_CLASSES,
                in_chans = in_channels
            )
            
            print(f"Random initialization: (PRETRAINED = False)")
    
    except Exception as e:
        raise ValueError(
            f"Model oluşturulamadı: {config.MODEL_NAME}\n"
            f"Hata: {str(e)}\n"
            f"Çözüm-1: PRETRAINED = False yapın config.py dosyasında\n"
            f"Çözüm-2: python scripts/download_weights_requests.py commandini terminalde çalıştırın"
        )
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel İstatistikleri:")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Model hazır!\n")
    
    return model

def get_model_info(model_name):
    """Kullanılan modelin bilgisini getirir."""
    try:
        model = timm.create_model(model_name, pretrained = False, num_classes = 1000)
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'name': model_name,
            'total_params': False,
            'available': True
            }
    except Exception as e:
        return {
            'name': model_name,
            'available': False,
            'error': str(e)
        }
    
def list_available_models():
    """Mevcut modelleri gösterir"""
    return timm.list_models()

# Test kodu
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import Config
    
    print("="*70)
    print("MODEL FACTORY TEST (LOCAL WEIGHTS)")
    print("="*70)
    
    # Test 1: Model oluştur
    print("\nTest 1: Model Olusturma")
    Config.PRETRAINED = True
    model = get_model(Config)

    # Test 2: Forward pass
    print("\nTest 2: Forward Pass")
    x = torch.randn(2, 2, 384, 384)  # Bilateral için 2 kanal
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Beklenen: (2, {Config.NUM_CLASSES})")
    
    assert output.shape == (2, Config.NUM_CLASSES), "Shape yanlis!"
    print("Test basarili!")
    
    print("\n" + "="*70)