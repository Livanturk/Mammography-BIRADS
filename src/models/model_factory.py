"""
Early Fusion
Config-aware model oluşturma
"""

import timm



def get_model(config):
    """
    Early fusion için model factory
    """
    
    # Bilateral için 2 kanal (CC + MLO)
    # Multi-view için 4 kanal (LCC + LMLO + RCC + RMLO)
    in_channels = 2 if config.APPROACH == "bilateral" else 4
    
    print(f"Model oluşturuluyor...")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Pretrained: {config.PRETRAINED}")
    print(f"In channels: {in_channels}")
    print(f"Num classes: {config.NUM_CLASSES}")
    
    try:
        model = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            num_classes=config.NUM_CLASSES,
            in_chans=in_channels
        )
    except Exception as e:
        raise ValueError(
            f"Model oluşturulamadı: {config.MODEL_NAME}\n"
            f"Hata: {str(e)}\n"
            f"timm.list_models() ile mevcut modelleri görebilirsin."
        )
    
    # Model bilgileri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model İstatistikleri:")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"✅ Model hazır!")
    
    return model


def list_available_models():
    """
    timm kütüphanesindeki mevcut modelleri listele
    
    Örnek:
        >>> models = list_available_models()
        >>> print(f"Toplam {len(models)} model mevcut")
        >>> print("İlk 10 model:", models[:10])
    """
    return timm.list_models()


def get_model_info(model_name):
    """
    Bir model hakkında bilgi al
    
    Args:
        model_name: Model ismi (örn. 'efficientnet_b0')
    
    Returns:
        dict: Model bilgileri
    
    Example:
        >>> info = get_model_info('efficientnet_b0')
        >>> print(info)
    """
    try:
        # Model oluştur (pretrained=False, hızlı)
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        
        # Bilgileri topla
        total_params = sum(p.numel() for p in model.parameters())
        
        # Default input size (bazı modellerde var)
        try:
            default_cfg = model.default_cfg
            input_size = default_cfg.get('input_size', (3, 224, 224))
        except:
            input_size = (3, 224, 224)
        
        return {
            'name': model_name,
            'total_params': total_params,
            'input_size': input_size,
            'available': True
        }
    
    except Exception as e:
        return {
            'name': model_name,
            'available': False,
            'error': str(e)
        }


