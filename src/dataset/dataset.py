import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

class BilateralMammogramDataset(Dataset):
    """
    Bilateral yaklaşım için hazırlanan dataset classı
    
    config: src/config.py içinden parametreleri alacak config objesi yaratır
    class_folders: data/ klasörü altındaki klasörlere erişir {1: BI-RADS_2 vs.}
    transform: Albumentation içindeki transformlar için
    is_train: Train mi test seti mi onu belirtir.
    """
    def __init__(
        self,
        config,
        class_folders,
        transform = None,
        is_train = True,
    ):
        self.config = config
        self.data_root = Path(config.DATA_ROOT)
        self.class_folders = class_folders
        self.transform = transform
        self.is_train = is_train
        self.fusion_type = config.FUSION_TYPE
        self.samples = []
        
        self._prepare_dataset()
        
        # Imbalance tespiti ve oversampling burada kontrol edilir
        if is_train and self.config.USE_OVERSAMPLING:
            classes_to_oversample = self._detect_imbalanced_classes()
            if classes_to_oversample:
                self._apply_oversampling(classes_to_oversample)
        
        print(f" {'Train' if is_train else 'Test'} dataset: {len(self)} örnek ({self.fusion_type})")
        
    
    def _prepare_dataset(self):
        """
        Bu fonksiyon dataları içeren klasörlerimizi tarayıp datasetlerimizi hazırlar.
        """
        # Biz BIRADS 1-2-4-5 değerleriyle çalışıyoruz.
        # Ancak looplar 0 index mantığıyla çalıştığı için (birads -> index) dönüşümü yapmamız gerek, yoksa loop çalışmaz.
        # Loop içerisinde daha sonra orijinal skorlara döndürecek bir dönüşüm var, bu sayede doğru bir şekilde klasörlere geçebiliyoruz. 
        # 1 -> 0 | 2 -> 1 | 4 -> 2 | 5 -> 3
        birads_to_label = {1: 0, 2: 1, 4: 2, 5: 3}
        
        # Burada birads parametresi 0'dan başlıyor (loop), birads_to_label dictionary'si bunun için yapılıyor.
        # Bu loop ile data klasörü altında olan BI-RADS_1, BI-RADS_1_Test vs. klasörlerine iniyoruz.
        for birads, folder_name in self.class_folders.items():
            class_path = self.data_root / folder_name             # Örn: class_path = data / BI-RADS_2
            if not class_path.exists():
                print(f"{class_path} bulunamadı")
                continue
            
            # Bu kod ile indexe dönüştürdüğümüz birads skorları orijinal haline geri döner
            label = birads_to_label[birads] # Örn: birads_to_label[1] = 2 (Index 1)(BI-RADS 2)
            
            # BIRADS klasörlerindeki hastaları (85675434 gibi) alır
            # Ayrıca bu HastaID klasörlerinin arasına klasör olmayan veriler karışırsa veya HastaID olmayan klasör karışırsa, bunları datasete eklememesi için de kontrol ediyoruz.
            # HastaID'ler gördüğüm kadarıyla 9 rakamlı oluyor, o yüzden böyle bir filtre verdim, belki daha sonra iyileştirilebilir.
            patients = [f for f in class_path.iterdir() if f.is_dir() and len(f.name) == 9]
            
            print(f" BI-RADS {birads}: {len(patients)} hasta")
            
            # 8537534 gibi hasta klasörlerinin içinde döner 
            for patient_folder in patients:
                lcc = patient_folder / self.config.VIEW_NAMES[0] # 85674352/LCC.png gibi
                lmlo = patient_folder / self.config.VIEW_NAMES[1]
                rcc = patient_folder / self.config.VIEW_NAMES[2]
                rmlo = patient_folder / self.config.VIEW_NAMES[3]
                
                # bir hasta klasöründe 4 görüntünün de olup olmadığını kontrol eden if bloğu
                # modelin çalışabilmesi için 4 görüntüye de ihtiyacımız var çünkü.
                if all(p.exists() for p in [lcc, lmlo, rcc, rmlo]):           
                    # Sol memenin görüntülerini ayırır
                    self.samples.append({
                        'cc_path': str(lcc),
                        'mlo_path': str(lmlo),
                        'label': label,
                        'side': 'left',
                        'patient': patient_folder.name,
                        'birads': birads
                        })
                    
                    # Sağ memenin görüntüleri ayırır
                    self.samples.append({
                        'cc_path' : str(rcc),
                        'mlo_path': str(rmlo),
                        'label': label,
                        'side': 'right',
                        'patient': patient_folder.name,
                        'birads': birads
                    })
                    
    def _detect_imbalanced_classes(self) -> List[int]:
        """
        Verdiğimiz verisetinde imbalance olup olmadığı kontrol eder
        Eğer imbalance varsa, oversampling stratejisini (auto, threshold, manual) seçer (bunu config.py dosyasından kontrol edip değiştirebilirsiniz) 
        !! Bu fonksiyon oversampling'in nasıl yapılacağına dair bilgi içermiyor
        !! bir alttaki apply_oversampling fonksiyonu oversample işlemi (oversample metodları) yapar
        !! Bu fonksiyon sadece imbalance tespiti ve oversampling stratejisi seçimi içindir. 
        """
        class_counts = self.get_class_distribution()
        
        if not class_counts:
            return []
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        total_count = sum(class_counts.values())
        
        print(f"Class Distribution:")
        print(f"En yüksek: {max_count} örnek")
        print(f"En düşük: {min_count} örnek")
        print(f"Toplam: {total_count}")
        print(f"Imbalance oranı: %{min_count/max_count:.2f}")
        
        # Loopta kullanılacak parametre 0'dan başlıyor, o yüzden bu mappingi oluşturdum.
        birads_mapping = {0: 1, 1: 2, 2: 4, 3: 5}
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_count) * 100
            birads = birads_mapping[class_id] # Burda tekrardan orijinal birads skoruna döner
            print(f"BI-RADS {birads} (class {class_id}): {count:4d} örnek %({percentage:5.1f})")
        
        classes_to_oversample = []
        strategy = self.config.OVERSAMPLING_STRATEGY
        
        # Manual oversampling olursa bunu elle config dosyasındaki MANUAL_OVERSAMPLE_CLASSES'da ekleriz ve orada değiştirebiliriz
        if strategy == "manual":
            classes_to_oversample = self.config.MANUAL_OVERSAMPLE_CLASSES
            print(f" Manual oversampling: {classes_to_oversample}")
        
            '''
    ---OVERSAMPLING KULLANIM REHBERİ---
                
    Senaryo 1: Dengeli veri
        - Tüm sınıflar: ~200 örnek
        - THRESHOLD (0.5): Hiçbir şey yapılmaz (hepsi %50'nin üstünde)
        - AUTO: Hiçbir şey yapılmaz (hepsi mean civarında)
        - Sonuç: İkisi de doğru

    Senaryo 2: Tek sınıf çok az
        - BI-RADS 1,4,5: 1000 örnek, BI-RADS 2: 50 örnek
        - THRESHOLD (0.5): BI-RADS 2'yi çoğalt (500'den az)
        - AUTO: BI-RADS 2'yi çoğalt (mean-std'den çok az)
        - Sonuç: İkisi de doğru

    Senaryo 3: İki sınıf biraz az
        - BI-RADS 1: 1000, BI-RADS 2: 400, BI-RADS 4: 450, BI-RADS 5: 900
        - THRESHOLD (0.5): Hiçbiri çoğaltılmaz (400 ve 450, 500'ün altında değil)
        - THRESHOLD (0.4): Hiçbiri çoğaltılmaz (hepsi 400'ün üstünde)
        - AUTO: Mean=687, Std=254, Threshold=433 → BI-RADS 2 çoğaltılır
        - Sonuç: AUTO daha hassas

    Senaryo 4: Veri çok dengesiz
        - BI-RADS 1: 10000, BI-RADS 2: 100, BI-RADS 4: 200, BI-RADS 5: 300
        - THRESHOLD (0.5): Hepsi çoğaltılır (5000'den az)
        - AUTO: Mean=2650, Std=4300, Threshold=-1650 (negatif!) → Hepsi çoğaltılır
        - Sonuç: İkisi de çoğaltır ama AUTO'nun threshold'u anlamsız olabilir

    HANGİSİNİ KULLANMALI?
    
    THRESHOLD kullan eğer:
        - Veri setini iyi tanıyorsan
        - Spesifik bir kontrol istiyorsan
        - "En az %50 olsun" gibi net bir kriter varsa

    AUTO kullan eğer:
        - Veri setini ilk kez görüyorsan
        - İstatistiksel yaklaşım istiyorsan
        - Veri seti dinamik değişiyorsa
        - "Anormal derecede az olanları bul" istiyorsan
            '''
               
        # (bir classtaki örnek sayısı / en kalabalık class örnek sayısı) belirli bir thresholdun altındaysa o class oversample edilir.
        # Threshold değerini configden ayarlayabilirsiniz (OVERSAMPLING_THRESHOLD)
        elif strategy == "threshold":
            # en kalabalık sınıf x threshold oranı = sınır
            threshold_count = max_count * self.config.OVERSAMPLING_THRESHOLD
            
            for class_id, count in class_counts.items():
                # Eğer bir classın örnek sayısı threshold değerinden azsa oversample yapılacak class olarak eklenir.
                if count < threshold_count:
                    classes_to_oversample.append(class_id)
        
        # Auto da aslında threshold gibi çalışıyor.
        # İkisi arasındaki fark, threshold yaklaşımında sayısal bir sınırı geçip geçmemesine bakmak
        # Auto yaklaşımında da bir sınır var ancak bu sayısal bir sınırla değil, istatistiki bir sınırla (mean-std) belirleniyor.
        elif strategy == "auto":
            counts = list(class_counts.values())
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            threshold = mean_count - std_count
            
            for class_id, count in class_counts.items():
                if count < threshold:
                    classes_to_oversample.append(class_id) 
                    
            print(f"\nAuto oversampling:")
            print(f"Mean: {mean_count:.0f}")
            print(f"Std: {std_count:.0f}")
            print(f"Threshold: {threshold:.0f}")
            print(f"Classlar: {classes_to_oversample}")
        
        if not classes_to_oversample:
            print(f"\nImbalance tespit edilmedi")
        
        return classes_to_oversample    
                 
        
    def _apply_oversampling(self, classes_to_oversample: List[int]):
        """
        Oversample işlemi burada yapılıyor.
        Küçük verisestinde kabul edilebilir ancak genel olarak anlamsız bir yaklaşım olabilir
        Çünkü aynı görüntüden 5 tane üretmek modelin daha iyi öğrenmesini sağlamayabilir, aksine overfite sebep olabilir.
        Bu durumu augmentation yaparak biraz indirgeyebiliriz.
        Eğer veri seti büyükse oversampling yapmaması için config.py içerisinde USE_OVERSAMPLING variable'ını False yapın
        """
        if not classes_to_oversample:
            return
        
        original_count = len(self.samples)
        class_counts = self.get_class_distribution()
        max_count = max(class_counts.values())
        
        
        for class_id in classes_to_oversample:
            class_samples = [s for s in self.samples if s['label'] == class_id]
            
            if not class_samples:
                continue
            
            current_count = len(class_samples)
            multiplier = max_count // current_count
            
            additional_samples = class_samples * (multiplier - 1)
            self.samples.extend(additional_samples)
            
            birads_mapping = {0: 1, 1: 2, 2: 4, 3: 5}
            birads = birads_mapping[class_id]
            print(f"BI-RADS {birads}: {current_count} → {current_count * multiplier} (×{multiplier})")
        
        print(f"\n Sonuç: {original_count} → {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Örnek yüklemek için"""
        sample = self.samples[idx]
        
        cc_img = np.array(Image.open(sample['cc_path']).convert('L'))
        mlo_img = np.array(Image.open(sample['mlo_path']).convert('L'))
        
        if self.transform:
            cc_img = self.transform(image=cc_img)['image']
            mlo_img = self.transform(image=mlo_img)['image']
        
        label = sample['label']
        
        if self.fusion_type in ["late", "attention"]:
            return cc_img, mlo_img, label
        else:
            combined = torch.cat([cc_img, mlo_img], dim=0)
            return combined, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Sınıf dağılımını almamızı sağlar"""
        labels = [s['label'] for s in self.samples]
        return dict(Counter(labels))


class MultiViewDataset(Dataset):
    """Multi-view dataset (4 görüntü)"""
    
    def __init__(self, config, class_folders, transform=None, is_train=True):
        self.config = config
        self.data_root = Path(config.DATA_ROOT)
        self.class_folders = class_folders
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        self._prepare_dataset()
        
        print(f"{'Train' if is_train else 'Test'} dataset: {len(self)} örnek (multi-view)")
    
    def _prepare_dataset(self):
        """Klasörleri tara"""
        birads_to_label = {1: 0, 2: 1, 4: 2, 5: 3}
        
        for birads, folder_name in self.class_folders.items():
            class_path = self.data_root / folder_name
            if not class_path.exists():
                continue
            
            label = birads_to_label[birads]
            patients = [f for f in class_path.iterdir() 
                       if f.is_dir() and len(f.name) == 9]
            
            print(f"  BI-RADS {birads}: {len(patients)} hasta")
            
            for patient_folder in patients:
                images = {
                    view.split('.')[0]: patient_folder / view
                    for view in self.config.VIEW_NAMES
                }
                
                if all(p.exists() for p in images.values()):
                    self.samples.append({
                        'images': {k: str(v) for k, v in images.items()},
                        'label': label,
                        'patient': patient_folder.name,
                        'birads': birads
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        imgs = []
        for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
            img = np.array(Image.open(sample['images'][view]).convert('L'))
            if self.transform:
                img = self.transform(image=img)['image']
            imgs.append(img)
        
        combined = torch.stack(imgs, dim=0)
        label = sample['label']
        
        return combined, label
    
    def get_class_distribution(self):
        labels = [s['label'] for s in self.samples]
        return dict(Counter(labels))


def get_transforms(config, is_train=True, aggressive=False):
    """
    Albumentations transformları
    
    Duruma bağlı olarak agressive veya not agressive yapabiliriz.
    Default olarak False. Bunun ayarı fonksiyon parametresinden ayarlanıyor, config.py dosyasından değil.
    
    Sadece train verilerine uygulanır, test verilerine uygulanmaz.
    """
    if is_train:
        if aggressive:
            return A.Compose([
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
                A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.3
                ),
                A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
                ToTensorV2()
            ])
    else:
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            ToTensorV2()
        ])
        
