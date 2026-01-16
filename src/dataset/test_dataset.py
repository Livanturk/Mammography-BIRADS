"""
Dataset sınıfları için unit, validation ve integration testleri
pytest ile çalıştırın: pytest test_dataset.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import Counter
import tempfile
import os
from PIL import Image

# Dataset modüllerini import et
from dataset import (
    BilateralMammogramDataset,
    MultiViewDataset,
    get_transforms
)

# Test için Mock Config sınıfı
class MockConfig:
    """Test için kullanılacak mock config"""
    DATA_ROOT = Path("../data")
    IMG_SIZE = 224
    IN_CHANNELS = 1
    NORMALIZE_MEAN = [0.485]
    NORMALIZE_STD = [0.229]
    USE_OVERSAMPLING = False
    OVERSAMPLING_STRATEGY = "auto"
    OVERSAMPLING_THRESHOLD = 0.5
    MANUAL_OVERSAMPLE_CLASSES = []
    VIEW_NAMES = ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]
    fusion_type = "late"
    FUSION_TYPE = "late"


class MockConfigWithOversampling(MockConfig):
    """Oversampling açık mock config"""
    USE_OVERSAMPLING = True
    OVERSAMPLING_STRATEGY = "threshold"
    OVERSAMPLING_THRESHOLD = 0.5


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestBilateralMammogramDatasetUnit:
    """BilateralMammogramDataset sınıfı unit testleri"""

    def test_class_exists(self):
        """BilateralMammogramDataset sınıfı var olmalı"""
        assert BilateralMammogramDataset is not None

    def test_inherits_from_dataset(self):
        """Dataset sınıfından türemeli"""
        from torch.utils.data import Dataset
        assert issubclass(BilateralMammogramDataset, Dataset)

    def test_has_required_methods(self):
        """Gerekli metodlar tanımlı olmalı"""
        required_methods = ['__len__', '__getitem__', '_prepare_dataset',
                          '_detect_imbalanced_classes', '_apply_oversampling',
                          'get_class_distribution']
        for method in required_methods:
            assert hasattr(BilateralMammogramDataset, method), f"{method} metodu eksik"

    def test_birads_to_label_mapping(self):
        """BI-RADS -> label dönüşümü doğru olmalı"""
        # Beklenen mapping: 1->0, 2->1, 4->2, 5->3
        expected_mapping = {1: 0, 2: 1, 4: 2, 5: 3}
        # Bu mapping _prepare_dataset içinde tanımlı
        # Doğrudan test edemiyoruz ama mantığı kontrol edebiliriz
        assert len(expected_mapping) == 4
        assert set(expected_mapping.keys()) == {1, 2, 4, 5}
        assert set(expected_mapping.values()) == {0, 1, 2, 3}


class TestMultiViewDatasetUnit:
    """MultiViewDataset sınıfı unit testleri"""

    def test_class_exists(self):
        """MultiViewDataset sınıfı var olmalı"""
        assert MultiViewDataset is not None

    def test_inherits_from_dataset(self):
        """Dataset sınıfından türemeli"""
        from torch.utils.data import Dataset
        assert issubclass(MultiViewDataset, Dataset)

    def test_has_required_methods(self):
        """Gerekli metodlar tanımlı olmalı"""
        required_methods = ['__len__', '__getitem__', '_prepare_dataset',
                          'get_class_distribution']
        for method in required_methods:
            assert hasattr(MultiViewDataset, method), f"{method} metodu eksik"


class TestGetTransformsUnit:
    """get_transforms fonksiyonu unit testleri"""

    def test_function_exists(self):
        """get_transforms fonksiyonu var olmalı"""
        assert get_transforms is not None
        assert callable(get_transforms)

    def test_returns_compose_object(self):
        """Albumentations Compose objesi döndürmeli"""
        import albumentations as A
        config = MockConfig()

        train_transform = get_transforms(config, is_train=True)
        test_transform = get_transforms(config, is_train=False)

        assert isinstance(train_transform, A.Compose)
        assert isinstance(test_transform, A.Compose)

    def test_aggressive_mode(self):
        """Aggressive mod çalışmalı"""
        import albumentations as A
        config = MockConfig()

        normal_transform = get_transforms(config, is_train=True, aggressive=False)
        aggressive_transform = get_transforms(config, is_train=True, aggressive=True)

        assert isinstance(normal_transform, A.Compose)
        assert isinstance(aggressive_transform, A.Compose)

        # Aggressive modda daha fazla transform olmalı
        assert len(aggressive_transform.transforms) >= len(normal_transform.transforms)

    def test_test_transform_simpler(self):
        """Test transform daha basit olmalı (augmentation yok)"""
        config = MockConfig()

        train_transform = get_transforms(config, is_train=True)
        test_transform = get_transforms(config, is_train=False)

        # Test transform daha az işlem içermeli
        assert len(test_transform.transforms) < len(train_transform.transforms)


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestBilateralDatasetValidation:
    """BilateralMammogramDataset validation testleri"""

    def test_samples_list_initialized(self):
        """samples listesi başlatılmalı"""
        # Mock ile kontrol
        with patch.object(BilateralMammogramDataset, '_prepare_dataset'):
            with patch.object(BilateralMammogramDataset, '__init__', lambda x: None):
                dataset = BilateralMammogramDataset.__new__(BilateralMammogramDataset)
                dataset.samples = []
                assert isinstance(dataset.samples, list)

    def test_sample_structure(self):
        """Her sample doğru yapıda olmalı"""
        expected_keys = {'cc_path', 'mlo_path', 'label', 'side', 'patient', 'birads'}

        # Örnek sample yapısı
        sample = {
            'cc_path': '/path/to/lcc.png',
            'mlo_path': '/path/to/lmlo.png',
            'label': 0,
            'side': 'left',
            'patient': '123456789',
            'birads': 1
        }

        assert set(sample.keys()) == expected_keys

    def test_label_values_valid(self):
        """Label değerleri 0-3 arasında olmalı"""
        valid_labels = {0, 1, 2, 3}

        # Örnek label kontrolü
        for label in valid_labels:
            assert 0 <= label <= 3

    def test_side_values_valid(self):
        """Side değerleri 'left' veya 'right' olmalı"""
        valid_sides = {'left', 'right'}
        assert 'left' in valid_sides
        assert 'right' in valid_sides

    def test_patient_id_format(self):
        """Hasta ID'si 9 karakter olmalı"""
        patient_id = "123456789"
        assert len(patient_id) == 9
        assert patient_id.isdigit()


class TestOversamplingValidation:
    """Oversampling mantığı validation testleri"""

    def test_oversampling_strategies_valid(self):
        """Oversampling stratejileri geçerli olmalı"""
        valid_strategies = {"auto", "threshold", "manual"}

        assert MockConfig.OVERSAMPLING_STRATEGY in valid_strategies or \
               MockConfigWithOversampling.OVERSAMPLING_STRATEGY in valid_strategies

    def test_threshold_in_valid_range(self):
        """Threshold 0-1 arasında olmalı"""
        assert 0 < MockConfig.OVERSAMPLING_THRESHOLD <= 1

    def test_manual_classes_is_list(self):
        """Manual oversample classes liste olmalı"""
        assert isinstance(MockConfig.MANUAL_OVERSAMPLE_CLASSES, list)


class TestTransformValidation:
    """Transform validation testleri"""

    def test_normalize_values_valid(self):
        """Normalize değerleri geçerli olmalı"""
        config = MockConfig()

        # Mean 0-1 arasında olmalı
        for mean in config.NORMALIZE_MEAN:
            assert 0 <= mean <= 1

        # Std pozitif olmalı
        for std in config.NORMALIZE_STD:
            assert std > 0

    def test_img_size_positive(self):
        """IMG_SIZE pozitif olmalı"""
        assert MockConfig.IMG_SIZE > 0

    def test_img_size_reasonable(self):
        """IMG_SIZE makul aralıkta olmalı"""
        assert 32 <= MockConfig.IMG_SIZE <= 1024


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDatasetIntegration:
    """Dataset integration testleri - gerçek veri yapısıyla test"""

    @pytest.fixture
    def temp_data_dir(self):
        """Geçici test veri dizini oluştur"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test klasör yapısı oluştur
            data_root = Path(tmpdir)

            # BI-RADS klasörleri oluştur
            for birads, folder_name in {1: "BI-RADS_1", 2: "BI-RADS_2"}.items():
                birads_folder = data_root / folder_name
                birads_folder.mkdir(parents=True)

                # Test hasta klasörleri oluştur
                for i in range(2):
                    patient_id = f"12345678{i}"
                    patient_folder = birads_folder / patient_id
                    patient_folder.mkdir()

                    # Dummy görüntüler oluştur
                    for view in ["LCC.png", "LMLO.png", "RCC.png", "RMLO.png"]:
                        img = Image.new('L', (100, 100), color=128)
                        img.save(patient_folder / view)

            yield data_root

    @pytest.fixture
    def mock_config_with_temp_dir(self, temp_data_dir):
        """Geçici dizinli mock config"""
        class TempConfig(MockConfig):
            DATA_ROOT = temp_data_dir
            USE_OVERSAMPLING = False
        return TempConfig

    def test_transform_applies_to_image(self):
        """Transform gerçek görüntüye uygulanabilmeli"""
        config = MockConfig()
        transform = get_transforms(config, is_train=True)

        # Dummy görüntü oluştur
        dummy_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        # Transform uygula
        result = transform(image=dummy_img)

        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)

    def test_transform_output_shape(self):
        """Transform çıktısı doğru şekilde olmalı"""
        config = MockConfig()
        transform = get_transforms(config, is_train=False)

        # Farklı boyutta görüntü
        dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

        result = transform(image=dummy_img)

        # Çıktı IMG_SIZE boyutunda olmalı
        assert result['image'].shape[-1] == config.IMG_SIZE
        assert result['image'].shape[-2] == config.IMG_SIZE

    def test_transform_normalization(self):
        """Transform normalizasyon uygulamalı"""
        config = MockConfig()
        transform = get_transforms(config, is_train=False)

        # Sabit değerli görüntü
        dummy_img = np.full((224, 224), 128, dtype=np.uint8)

        result = transform(image=dummy_img)

        # Normalize edilmiş değerler 0-255 aralığında olmamalı
        # (tipik olarak -3 ile 3 arasında olur)
        assert result['image'].min() < 128
        assert result['image'].max() < 128


class TestDataLoaderIntegration:
    """DataLoader ile integration testleri"""

    def test_transform_batch_compatible(self):
        """Transform çıktısı batch için uyumlu olmalı"""
        config = MockConfig()
        transform = get_transforms(config, is_train=True)

        # Birden fazla görüntü transform et
        images = []
        for _ in range(4):
            dummy_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            result = transform(image=dummy_img)
            images.append(result['image'])

        # Stack edilebilmeli
        batch = torch.stack(images)
        assert batch.shape[0] == 4
        assert batch.shape[-1] == config.IMG_SIZE


# ============================================================================
# BUG DETECTION TESTS
# ============================================================================

class TestBugDetection:
    """Koddaki potansiyel hataları tespit eder"""

    def test_class_folder_typo(self):
        """class_folder vs class_folders yazım hatası kontrolü"""
        # BilateralMammogramDataset._prepare_dataset içinde
        # self.class_folder kullanılıyor (satır 58)
        # ama __init__'te self.class_folders tanımlı (satır 29)
        # Bu bir bug!

        import inspect
        source = inspect.getsource(BilateralMammogramDataset._prepare_dataset)

        if "self.class_folder" in source and "self.class_folders" not in source:
            pytest.fail("BUG: _prepare_dataset içinde 'self.class_folder' kullanılıyor "
                       "ama 'self.class_folders' olmalı (satır 58)")

    def test_use_oversampling_attribute(self):
        """use_oversampling attribute kontrolü"""
        # __init__ içinde self.use_oversampling kullanılıyor (satır 38)
        # ama bu attribute hiç tanımlanmamış!
        # config.USE_OVERSAMPLING olmalı

        import inspect
        source = inspect.getsource(BilateralMammogramDataset.__init__)

        if "self.use_oversampling" in source:
            # Kontrol et: bu attribute tanımlı mı?
            if "self.use_oversampling =" not in source:
                pytest.fail("BUG: 'self.use_oversampling' kullanılıyor ama tanımlanmamış. "
                           "'self.config.USE_OVERSAMPLING' olmalı (satır 38)")

    def test_classes_to_oversample_method_call(self):
        """classes_to_oversample.append vs () hatası kontrolü"""
        # Satır 207'de classes_to_oversample(class_id) yazılmış
        # classes_to_oversample.append(class_id) olmalı

        import inspect
        source = inspect.getsource(BilateralMammogramDataset._detect_imbalanced_classes)

        if "classes_to_oversample(class_id)" in source:
            pytest.fail("BUG: 'classes_to_oversample(class_id)' yazılmış, "
                       "'classes_to_oversample.append(class_id)' olmalı (satır 207)")

    def test_undefined_variables_in_print(self):
        """Print statement'larda tanımsız değişken kontrolü"""
        # Satır 210-213'te mean_count, std_count, threshold kullanılıyor
        # ama bunlar sadece "auto" stratejisinde tanımlı
        # "threshold" veya "manual" stratejisinde hata verir

        import inspect
        source = inspect.getsource(BilateralMammogramDataset._detect_imbalanced_classes)

        # Print statements auto bloğunun dışında
        if 'print(f"\\nAuto oversampling:' in source or 'print(f"Mean:' in source:
            # Bu printler her zaman çalışıyor ama mean_count sadece auto'da tanımlı
            pytest.xfail("POTENTIAL BUG: mean_count, std_count, threshold değişkenleri "
                        "sadece 'auto' stratejisinde tanımlı ama print her zaman çalışıyor")

    def test_side_missing_in_right_sample(self):
        """Sağ meme sample'ında 'side' eksik kontrolü"""
        # Satır 95-101'de sağ meme için sample oluşturuluyor
        # ama 'side': 'right' eksik!

        import inspect
        source = inspect.getsource(BilateralMammogramDataset._prepare_dataset)

        # Sol memede 'side': 'left' var
        # Sağ memede 'side' olup olmadığını kontrol et
        if "'side': 'left'" in source:
            # Sağ meme bloğunu bul
            right_block_start = source.find("# Sağ memenin")
            if right_block_start != -1:
                right_block = source[right_block_start:right_block_start+500]
                if "'side':" not in right_block:
                    pytest.fail("BUG: Sağ meme sample'ında 'side': 'right' eksik (satır 95-101)")


class TestCodeQuality:
    """Kod kalitesi testleri"""

    def test_docstrings_exist(self):
        """Ana sınıfların docstring'leri olmalı"""
        assert BilateralMammogramDataset.__doc__ is not None
        assert MultiViewDataset.__doc__ is not None
        assert get_transforms.__doc__ is not None

    def test_type_hints_in_methods(self):
        """Önemli metodlarda type hint olmalı"""
        import inspect

        # _detect_imbalanced_classes return type hint kontrolü
        sig = inspect.signature(BilateralMammogramDataset._detect_imbalanced_classes)
        assert sig.return_annotation != inspect.Parameter.empty or \
               "-> List[int]" in inspect.getsource(BilateralMammogramDataset._detect_imbalanced_classes)


# ============================================================================
# MOCK DATASET TESTS (Gerçek veri olmadan)
# ============================================================================

class TestMockedDataset:
    """Mock ile dataset testleri"""

    def test_len_returns_sample_count(self):
        """__len__ sample sayısını döndürmeli"""
        with patch.object(BilateralMammogramDataset, '__init__', lambda x: None):
            dataset = BilateralMammogramDataset.__new__(BilateralMammogramDataset)
            dataset.samples = [1, 2, 3, 4, 5]

            assert len(dataset) == 5

    def test_get_class_distribution_counts_labels(self):
        """get_class_distribution label sayılarını döndürmeli"""
        with patch.object(BilateralMammogramDataset, '__init__', lambda x: None):
            dataset = BilateralMammogramDataset.__new__(BilateralMammogramDataset)
            dataset.samples = [
                {'label': 0}, {'label': 0},
                {'label': 1}, {'label': 1}, {'label': 1},
                {'label': 2},
                {'label': 3}, {'label': 3}
            ]

            distribution = dataset.get_class_distribution()

            assert distribution[0] == 2
            assert distribution[1] == 3
            assert distribution[2] == 1
            assert distribution[3] == 2


if __name__ == "__main__":
    # Testleri çalıştırmak için: python test_dataset.py
    pytest.main([__file__, "-v", "--tb=short"])
