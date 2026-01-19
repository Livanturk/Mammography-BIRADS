"""
Dataset testleri
pytest -v tests/test_dataset.py
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from dataset.dataset import BilateralMammogramDataset, MultiViewDataset, get_transforms
from config import Config


class TestGetTransforms:
    """Transform testleri"""
    
    def test_train_transform_returns_tensor(self):
        """Train transform tensor döndürüyor mu?"""
        transform = get_transforms(Config, is_train=True)
        
        # Dummy grayscale image
        img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
        
        result = transform(image=img)
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
    
    def test_test_transform_returns_tensor(self):
        """Test transform tensor döndürüyor mu?"""
        transform = get_transforms(Config, is_train=False)
        
        img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
        
        result = transform(image=img)
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
    
    def test_transform_output_shape(self):
        """Transform sonucu doğru shape'te mi?"""
        transform = get_transforms(Config, is_train=False)
        
        img = np.random.randint(0, 255, (500, 600), dtype=np.uint8)
        
        result = transform(image=img)
        tensor = result['image']
        
        # (C, H, W) formatında olmalı
        assert tensor.shape == (1, Config.IMG_SIZE, Config.IMG_SIZE), \
            f"Beklenen: (1, {Config.IMG_SIZE}, {Config.IMG_SIZE}), Gelen: {tensor.shape}"
    
    def test_aggressive_transform(self):
        """Aggressive transform çalışıyor mu?"""
        transform = get_transforms(Config, is_train=True, aggressive=True)
        
        img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
        
        result = transform(image=img)
        assert isinstance(result['image'], torch.Tensor)
    
    def test_transform_normalization(self):
        """Normalizasyon uygulanıyor mu?"""
        transform = get_transforms(Config, is_train=False)

        # Tüm beyaz görüntü (255)
        img = np.full((384, 384), 255, dtype=np.uint8)

        result = transform(image=img)
        tensor = result['image']

        # Normalize edilmiş olmalı (0-1 arası değil, mean/std ile normalize)
        # ImageNet normalization: (1.0 - mean) / std ile >1 değerler üretir
        assert tensor.max() > 1.0  # Normalize sonrası >1 olabilir


class TestBilateralMammogramDataset:
    """BilateralMammogramDataset testleri"""
    
    def test_dataset_initialization_without_data(self):
        """Dataset başlatılabiliyor mu? (veri olmasa bile)"""
        # Geçici config (var olmayan klasör)
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = False
        
        # Hata vermemeli, sadece uyarı
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        assert isinstance(dataset, BilateralMammogramDataset)
        assert len(dataset) == 0  # Veri olmadığı için boş
    
    def test_get_class_distribution(self):
        """get_class_distribution metodu çalışıyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = False
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        dist = dataset.get_class_distribution()
        assert isinstance(dist, dict)
    
    def test_dataset_samples_structure(self):
        """Dataset samples yapısı doğru mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = False
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        # Samples bir liste olmalı
        assert isinstance(dataset.samples, list)


class TestMultiViewDataset:
    """MultiViewDataset testleri"""
    
    def test_dataset_initialization(self):
        """MultiView dataset başlatılabiliyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
        
        dataset = MultiViewDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        assert isinstance(dataset, MultiViewDataset)
        assert len(dataset) == 0


class TestOversamplingLogic:
    """Oversampling mantığı testleri"""
    
    def test_detect_imbalanced_classes_auto(self):
        """Auto oversampling detection çalışıyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1", 2: "BI-RADS_2"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = True
            OVERSAMPLING_STRATEGY = "auto"
            OVERSAMPLING_THRESHOLD = 0.5
            MANUAL_OVERSAMPLE_CLASSES = []
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        # Manuel samples ekleyelim
        dataset.samples = [
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '001', 'birads': 1},
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '002', 'birads': 1},
            {'label': 1, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '003', 'birads': 2},
        ]
        
        # Detection
        classes_to_oversample = dataset._detect_imbalanced_classes()
        
        # Class 1 (BI-RADS 2) az olduğu için tespit edilmeli
        assert isinstance(classes_to_oversample, list)
    
    def test_detect_imbalanced_classes_threshold(self):
        """Threshold oversampling detection çalışıyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1", 2: "BI-RADS_2"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = True
            OVERSAMPLING_STRATEGY = "threshold"
            OVERSAMPLING_THRESHOLD = 0.5
            MANUAL_OVERSAMPLE_CLASSES = []
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        dataset.samples = [
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '001', 'birads': 1},
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '002', 'birads': 1},
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '003', 'birads': 1},
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '004', 'birads': 1},
            {'label': 1, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '005', 'birads': 2},
        ]
        
        classes_to_oversample = dataset._detect_imbalanced_classes()
        
        # Class 1, class 0'ın %50'sinden az (1 < 2) → oversample edilmeli
        assert 1 in classes_to_oversample
    
    def test_detect_imbalanced_classes_manual(self):
        """Manuel oversampling çalışıyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1", 2: "BI-RADS_2"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = True
            OVERSAMPLING_STRATEGY = "manual"
            OVERSAMPLING_THRESHOLD = 0.5
            MANUAL_OVERSAMPLE_CLASSES = [1]  # Sadece class 1
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        dataset.samples = [
            {'label': 0, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '001', 'birads': 1},
            {'label': 1, 'cc_path': 'a', 'mlo_path': 'b', 'side': 'left', 'patient': '002', 'birads': 2},
        ]
        
        classes_to_oversample = dataset._detect_imbalanced_classes()
        
        # Manuel olarak [1] belirtildi
        assert classes_to_oversample == [1]
    
    def test_apply_oversampling(self):
        """Oversampling uygulanıyor mu?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1", 2: "BI-RADS_2"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = False  # Manuel test için
            OVERSAMPLING_STRATEGY = "manual"
            OVERSAMPLING_THRESHOLD = 0.5
            MANUAL_OVERSAMPLE_CLASSES = []
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=None,
            is_train=True
        )
        
        # Manuel samples
        dataset.samples = [
            {'label': 0, 'cc_path': 'a1', 'mlo_path': 'b1', 'side': 'left', 'patient': '001', 'birads': 1},
            {'label': 0, 'cc_path': 'a2', 'mlo_path': 'b2', 'side': 'left', 'patient': '002', 'birads': 1},
            {'label': 0, 'cc_path': 'a3', 'mlo_path': 'b3', 'side': 'left', 'patient': '003', 'birads': 1},
            {'label': 0, 'cc_path': 'a4', 'mlo_path': 'b4', 'side': 'left', 'patient': '004', 'birads': 1},
            {'label': 1, 'cc_path': 'c1', 'mlo_path': 'd1', 'side': 'left', 'patient': '005', 'birads': 2},
        ]
        
        original_count = len(dataset.samples)
        
        # Class 1'i oversample et
        dataset._apply_oversampling([1])
        
        # Artmış olmalı
        assert len(dataset.samples) > original_count
        
        # Class 1 örnek sayısı artmış olmalı
        class_1_count = sum(1 for s in dataset.samples if s['label'] == 1)
        assert class_1_count > 1


class TestDatasetGetItem:
    """__getitem__ testleri"""
    
    def test_getitem_structure_late_fusion(self):
        """Late fusion için __getitem__ doğru yapıda mı?"""
        class TempConfig:
            DATA_ROOT = Path("/nonexistent/path")
            TRAIN_CLASSES = {1: "BI-RADS_1"}
            VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
            FUSION_TYPE = "late"
            fusion_type = "late"
            USE_OVERSAMPLING = False
            IMG_SIZE = 384
            NORMALIZE_MEAN = [0.485]
            NORMALIZE_STD = [0.229]
        
        dataset = BilateralMammogramDataset(
            config=TempConfig,
            class_folders=TempConfig.TRAIN_CLASSES,
            transform=get_transforms(TempConfig, is_train=False),
            is_train=False
        )
        
        # Eğer samples boşsa, manuel ekle (gerçek dosya olmadan test)
        # Bu test gerçek veri olmadan çalışmayacak, skip edebiliriz
        pytest.skip("Gerçek veri olmadan test edilemez")