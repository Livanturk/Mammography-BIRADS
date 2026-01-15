"""
Bütün projenin konfigürasyon ayarları burada bulunuyor
Yeni bir experiment veya fine-tuning için buradaki parametreleri değiştirebilirsiniz
"""

import torch
from pathlib import Path

class Config:
    # Veri Yolu -  Proje yapısına göre burası update edilmeli (dataset klasörü config.py dosyasına göre nerede kalıyor?)
    DATA_ROOT = Path("../data")
    
    # Train
    TRAIN_CLASSES = {1: "BI-RADS_1", 2: "BI-RADS_2", 4: "BI-RADS_4", 5: "BI-RADS_5"}
    # Test
    TEST_CLASSES = {1: "BI-RADS_1_TEST", 2: "BI-RADS_2_TEST", 4: "BI-RADS_4_TEST", 5: "BI-RADS_5_TEST"}
    VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
    
    # STRATEJİ VE YAKLAŞIMLAR (APPROACH)
    """
    - "bilateral": Sağ-sol meme ayrı olacak şekile
    - "multi-view": 4 görüntü (RCC, LCC, RMLO, LMLO) birlikte verilir (Ama bunun yerine bilateral kullanacağız best practice olduğu için)
    """
    APPROACH = "bilateral"
    
    # FUSION TIPI (FUSION_TYPE)
    """
    - "early": 2 görüntüyü başta birleştirir
    - "late": Her görüntü ayrı CNN'den geçer => Önerilen
    - "attention": Late fusion + attention mechanism
    """
    FUSION_TYPE = "attention"
    
    # GÖRÜNTÜ
    IMG_SIZE = 384
    NUM_CLASSES = 4
    # ImageNet 3 channel bekliyor ancak elimizdeki görüntüler grayscale (1 kanallı)
    # Bu yüzden IN_CHANNELS ile modelin weightleri RGB channelların ortalaması alınarak 1 channela indirgeniyor.
    IN_CHANNELS = 1 # GÖrüntüler grayscale (tek kanallı) ise
    
    # Grayscale için NORMALİZASYON değerleri
    NORMALIZE_MEAN = [0.485]
    NORMALIZE_STD = [0.229]
    
    # TRAINING
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    PATIENCE = 10
    
    # Transfer Learning modelleri bazlı Learning Rateler
    """
    Modellerin kendilerine ait optimum learning rateleri var (araştırılmalı)
    Farklı bir model denemek istediğinizde aşağıdaki dictionary içine ('model ismi': learning_rate) şeklinde ekleyebilirsiniz
    """
    MODEL_LEARNING_RATES = {
        'efficientnet_b0': 1e-4,
        'resnet50': 1e-4,
        'convnext_tiny': 1e-5,
        'swin_tiny_patch4_window7_224': 1e-5,
        'vit_small_patch16_224': 1e-5,
    }
    
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    
    @classmethod
    def get_learning_rate(cls, model_name):
        """Kullanılan Transfer Learning modelinin learning rate değerini döndürür"""
        return cls.MODEL_LEARNING_RATES.get(model_name, 1e-4) # Verilen modeli MODEL_LEARNING_RATE içinde bulamazsa default olarak 1e-4 learning rate değerini atar.
    
    # Loss ve Optimizasyon
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING_FACTOR = 0.1 # %10 smoothing
    
    USE_CLASS_WEIGHTS = True # Class imbalance için
    
    # Oversampling Ayarları
    USE_OVERSAMPLING = True
    OVERSAMPLING_STRATEGY = "auto" # auto, threshold veya manual
    OVERSAMPLING_THRESHOLD = 0.5 # Eğer bir class, en fazla örnek olan sınıftan %X azsa, o class oversample edilir (Default %50, değiştirebilirsiniz)
    
    # Manual oversampling -> Eğer oversampling yaklaşımı manuel olacaksa (mesela sadece BI-RADS_2 oversample etmek istersek)
    MANUAL_OVERSAMPLE_CLASSES = [] # Mesela [1] olursa bu BI-RADS 2 olur (0 index mantığı var)
    
    # SCHEDULER
    SCHEDULER_TYPE = True
    
    # ATTENTION
    USE_ATTENTION = True
    
    # GPU-CPU Ayarı
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4 
    
    # MLFLOW Ayarları
    MLFLOW_TRACKING_URI = "./mlruns"
    MLFLOW_EXPERIMENT_NAME = "mamografi_birads"
    
    # CHECKPOINT
    CHECKPOINT_DIR = Path("./checkpoints")
    
    # SEED
    SEED = 42
    
    # GRAD-CAM
    USE_GRADCAM = True
    GRADCAM_SAMPLES = 10
    GRADCAM_DIR = Path("./gradcam_outputs")
    
    @classmethod
    def print_config(cls):
        print("\n" + "="*70)
        print("PROJE KONFİGÜRASYONU")
        print("="*70)
        print(f"Veri: {cls.DATA_ROOT}")
        print(f"Yaklaşım: {cls.APPROACH}")
        print(f"Fusion: {cls.FUSION_TYPE}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Learning Rate: {cls.get_learning_rate(cls.MODEL_NAME)}")
        print(f"Label Smoothing: {cls.USE_LABEL_SMOOTHING}")
        print(f"Class Weights: {cls.USE_CLASS_WEIGHTS} (otomatik)")
        print(f"Oversampling: {cls.USE_OVERSAMPLING} (strategy={cls.OVERSAMPLING_STRATEGY})")
        print(f"Attention: {cls.USE_ATTENTION}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Scheduler: {cls.SCHEDULER_TYPE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Grad-CAM: {cls.USE_GRADCAM}")
        print("="*70 + "\n")
        
        
class FastTestConfig(Config):
    """Hızlı test için (2 epoch)"""
    NUM_EPOCHS = 2
    BATCH_SIZE = 4
    USE_GRADCAM = False
    USE_OVERSAMPLING = False
    
class SmallDataConfig(Config):
    """Küçük veri seti için kullanılacak"""
    DATA_ROOT = Path("../data")
    MLFLOW_EXPERIMENT_NAME = "mamografi_birads_prototype_60mb"
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    OVERSAMPLING_STRATEGY = "auto"
    
class LargeDataConfig(Config):
    """Gerçek veri için"""
    DATA_ROOT = Path("../data")
    MLFLOW_EXPERIMENT_NAME = "mamografi_birads_production"
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    USE_CLASS_WEIGHTS = True
    USE_OVERSAMPLING = True
    OVERSAMPLING_STRATEGY = "threshold"  # Threshold-based
    OVERSAMPLING_THRESHOLD = 0.5  # %50'den az olan sınıfları çoğalt
    PATIENCE = 15


class SwinConfig(Config):
    """Swin Transformer için özel config (Buna ekstradan bakmak lazım biraz gelişigüzel parametre verdim)"""
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    BATCH_SIZE = 12
    NUM_EPOCHS = 70
    
    
