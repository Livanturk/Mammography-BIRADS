"""
Pretrained weights indirme - Manuel URL'ler + SSL bypass
pip install requests tqdm
python scripts/download_weights_requests.py
"""

import requests
from pathlib import Path
from tqdm import tqdm

# SSL uyarılarını kapat
requests.packages.urllib3.disable_warnings()

# Weights klasörü
WEIGHTS_DIR = Path(__file__).parent.parent / "pretrained_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# İndirme linkleri (Manuel URL'ler)
WEIGHTS_URLS = {
    'efficientnet_b0': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'convnext_tiny': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
    'swin_tiny_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
}


def download_file(url, save_path):
    """
    Dosya indir (progress bar + SSL bypass)
    
    Args:
        url: İndirme linki
        save_path: Kayıt yolu
    
    Returns:
        bool: Başarılı mı?
    """
    try:
        print(f"\n📥 İndiriliyor: {save_path.name}")
        print(f"   URL: {url}")
        
        # HEAD request ile dosya boyutunu al
        response = requests.head(url, verify=False, allow_redirects=True, timeout=10)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size > 0:
            print(f"   Boyut: {total_size / (1024*1024):.2f} MB")
        
        # Dosyayı indir
        response = requests.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        
        # Progress bar ile kaydet
        with open(save_path, 'wb') as f:
            if total_size > 0:
                with tqdm(
                    desc=f"   İndiriliyor",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Boyut bilinmiyorsa progress bar olmadan indir
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Dosya boyutunu kontrol et
        file_size = save_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ İndirildi: {file_size:.2f} MB")
        
        return True
    
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout hatası: Bağlantı zaman aşımına uğradı")
        return False
    
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Bağlantı hatası: İnternet bağlantısını kontrol et")
        return False
    
    except requests.exceptions.HTTPError as e:
        print(f"   ❌ HTTP hatası: {e}")
        return False
    
    except Exception as e:
        print(f"   ❌ Hata: {str(e)}")
        return False


def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("🔽 PRETRAINED WEIGHTS İNDİRME")
    print("   SSL Bypass + Progress Bar")
    print("="*70)
    print(f"\n📁 Klasör: {WEIGHTS_DIR.absolute()}")
    print(f"📊 Model sayısı: {len(WEIGHTS_URLS)}")
    
    success = 0
    failed = 0
    failed_models = []
    
    for model_name, url in WEIGHTS_URLS.items():
        save_path = WEIGHTS_DIR / f"{model_name}.pth"
        
        # Zaten varsa atla
        if save_path.exists():
            file_size = save_path.stat().st_size / (1024 * 1024)
            print(f"\n⏭️  Atlanıyor: {model_name}.pth (zaten var, {file_size:.2f} MB)")
            success += 1
            continue
        
        # İndir
        if download_file(url, save_path):
            success += 1
        else:
            failed += 1
            failed_models.append(model_name)
    
    # Özet
    print("\n" + "="*70)
    print("📊 ÖZET")
    print("="*70)
    print(f"✅ Başarılı: {success}/{len(WEIGHTS_URLS)}")
    print(f"❌ Başarısız: {failed}/{len(WEIGHTS_URLS)}")
    
    if failed_models:
        print(f"\n❌ Başarısız modeller:")
        for m in failed_models:
            print(f"   - {m}")
        print(f"\nİpucu: Başarısız modelleri manuel indirebilirsin:")
        for m in failed_models:
            print(f"   {WEIGHTS_URLS[m]}")
    
    # İndirilen dosyalar
    print(f"\n✅ İndirilen dosyalar:")
    total_size = 0
    for model_name in WEIGHTS_URLS.keys():
        weight_path = WEIGHTS_DIR / f"{model_name}.pth"
        if weight_path.exists():
            size = weight_path.stat().st_size / (1024 * 1024)
            total_size += size
            print(f"   ✅ {model_name}.pth ({size:.2f} MB)")
        else:
            print(f"   ❌ {model_name}.pth (eksik)")
    
    print(f"\n📦 Toplam boyut: {total_size:.2f} MB")
    print(f"📁 Konum: {WEIGHTS_DIR.absolute()}")
    print("\n" + "="*70)
    print("✅ Tamamlandı!")
    print("="*70)


if __name__ == "__main__":
    main()