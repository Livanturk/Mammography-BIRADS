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
    is_train: Train mi test seti mi onu ayırır.
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
        self.fusion_type = config.fusion_type
        self.samples = []
        
        self._prepare_dataset()
        
        # Imbalance tespiti ve oversampling burada kontrol edilir
        if is_train and self.use_oversampling:
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
        for birads, folder_name in self.class_folder.items():
            class_path = self.data_root / folder_name             # Örn: class_path = data / BI-RADS_2
            if not class_path.exists():
                print(f"{class_path} bulunamadı")
                continue
            
            # Bu kod ile indexe dönüştürdüğümüz birads skorları orijinal haline geri döner
            label = birads_to_label[birads] # Örn: birads_to_label[1] = 2 (Index 1)(BIRADS 2)
            
            # BIRADS klasörlerindeki hastaları (85675434 gibi) alır ve almadan önce klasör olup olmamasını kontrol eder.
            patients = [f for f in class_path.iterdir() if f.is_dir()]
            
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
        
        # Manual olursa bunu elle config dosyasındaki MANUAL_OVERSAMPLE_CLASSES'a bağlarız
        if strategy == "manual":
            classes_to_oversample = self.config.MANUAL_OVERSAMPLE_CLASSES
            print(f" Manual oversampling: {classes_to_oversample}")
        
        #  (bir classtaki örnek sayısı / en kalabalık class örnek sayısı) belirli bir thresholdun altındaysa o class oversample edilir.
        # Threshold değerini configden ayarlayabilirsiniz (OVERSAMPLING_THRESHOLD)
        elif strategy == "threshold":
            # en kalabalık sınıf x threshold oranı = sınır
            threshold_count = max_count * self.config.OVERSAMPLING_THRESHOLD
            
            for class_id, count in class_counts.items():
                # Eğer bir classın örnek sayısı threshold değerinden azsa oversample yapılacak class olarak eklenir.
                if count < threshold_count:
                    classes_to_oversample.append(class_id)
        
                    

            
            
        
        
        
