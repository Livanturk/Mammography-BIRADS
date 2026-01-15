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
                        'side': 'right',
                        'patient': patient_folder.name,
                        'birads': birads
                    })
                    
                