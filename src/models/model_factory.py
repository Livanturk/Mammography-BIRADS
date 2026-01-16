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
        new_conv = nn.Conv2D(
            in_channels,
            old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride = old_conv.stride,
            padding = old_conv.stride,
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
        
        model.conv_stem
    
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
                    new_conv.weight[:, i] = old_conv[:, i % 3]
        model.conv1 = new_conv
        
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
                    new_conv.weight[:, i] = old_conv[:, i % 3]
                    
            