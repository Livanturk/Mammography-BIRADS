"""
Pretrained weight kontrolü
"""

import torch
from pathlib import Path
import sys

# src klasörünü ekleme
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config
from models.model_factory import get_model

WEIGHTS_DIR =Path(__file__).parent.parent / "pretrained_weights"

# Test edilecek modeller ve image size'ları
MODELS_TO_TEST = {
    'efficientnet_b0': 384,
    'resnet50': 384,
    'convnext_tiny': 384,
    'swin_tiny_patch4_window7_224': 224,
}

def test_weight_file(model_name, img_size=384):
    """
    Bir model için weight dosyası kontrolü

    Input:
        - Model ismi
        - Image size (default: 384)
    Output:
        - Test sonuçları
    """
    result = {
        'model': model_name,
        'file_exists': False,
        'file_size_mb': 0,
        'loadable':  False,
        'model_creates': False,
        'forward_pass': False,
        'error': None
    }

    # Dosya kontrolü
    weight_path = WEIGHTS_DIR / f"{model_name}.pth"
    result['file_exists'] = weight_path.exists()

    if not result['file_exists']:
        result['error'] = "Dosya bulunamadı"
        return result

    # Dosya boyutu
    result['file_size_mb'] = weight_path.stat().st_size / (1024 * 1024)

    # Yüklenebilirlik durumu
    try:
        state_dict = torch.load(weight_path, map_location = 'cpu')
        result['loadable'] = True
    except Exception as e:
        result['error'] = f"Yüklenemedi: {e}"
        return result

    # Model Oluşuturabilme kontrolü
    try:
        # temp config
        class TempConfig:
            MODEL_NAME = model_name
            PRETRAINED = True
            NUM_CLASSES = 4
            APPROACH = 'bilateral'
            IMG_SIZE = img_size
            IN_CHANNELS = 1

        model = get_model(TempConfig)
        result['model_creates'] = True

    except Exception as e:
        result['error'] = f" Model oluşturulamadı: {str(e)}"
        return result

    # Forward pass çalışma kontrlü
    try:
        model.eval()
        dummy_input = torch.randn(1, 2, img_size, img_size)

        with torch.no_grad():
            output = model(dummy_input)

        if output.shape == (1, 4):
            result['forward_pass'] = True

        else:
            result['error'] = f"Output shape yanlış: {output.shape}"

    except Exception as e:
        result['error'] = f"Forward pass hatası: {str(e)}"
        return result

    return result


def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("🔍 PRETRAINED WEIGHTS TEST")
    print("="*70)
    print(f"\n📁 Weights klasörü: {WEIGHTS_DIR.absolute()}")
    print(f"📊 Test edilecek model sayısı: {len(MODELS_TO_TEST)}\n")
    
    results = []

    for model_name, img_size in MODELS_TO_TEST.items():
        print(f"\n{'='*70}")
        print(f"🧪 {model_name} (img_size: {img_size})")
        print('='*70)

        result = test_weight_file(model_name, img_size)
        results.append(result)
        
        # Sonuçları yazdır
        print(f"   📄 Dosya var: {'✅' if result['file_exists'] else '❌'}")
        
        if result['file_exists']:
            print(f"   💾 Boyut: {result['file_size_mb']:.2f} MB")
        
        print(f"   📥 Yüklenebilir: {'✅' if result['loadable'] else '❌'}")
        print(f"   🧠 Model oluşur: {'✅' if result['model_creates'] else '❌'}")
        print(f"   ▶️ Forward pass: {'✅' if result['forward_pass'] else '❌'}")
        
        if result['error']:
            print(f"   ⚠️ Hata: {result['error']}")
    
    # Özet
    print("\n" + "="*70)
    print("📊 ÖZET")
    print("="*70)
    
    success_count = sum(1 for r in results if r['forward_pass'])
    total_count = len(results)
    
    print(f"\n✅ Başarılı: {success_count}/{total_count}")
    print(f"❌ Başarısız: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 Tüm modeller çalışıyor!")
    else:
        print("\n⚠️ Bazı modeller çalışmıyor. Detaylar yukarıda.")
        
        failed = [r['model'] for r in results if not r['forward_pass']]
        print(f"\nBaşarısız modeller:")
        for model in failed:
            print(f"   - {model}")
    
    # Toplam boyut
    total_size = sum(r['file_size_mb'] for r in results if r['file_exists'])
    print(f"\n📦 Toplam boyut: {total_size:.2f} MB")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()