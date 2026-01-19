"""
Config testleri
pytest -v tests/test_config.py
"""

import pytest
import torch
from config import Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig


class TestConfig:
    """Config sınıfı testleri"""
    
    def test_config_attributes(self):
        """Config'in gerekli attribute'ları var mı?"""
        assert hasattr(Config, 'DATA_ROOT')
        assert hasattr(Config, 'IMG_SIZE')
        assert hasattr(Config, 'BATCH_SIZE')
        assert hasattr(Config, 'NUM_EPOCHS')
        assert hasattr(Config, 'MODEL_NAME')
        assert hasattr(Config, 'NUM_CLASSES')
        assert hasattr(Config, 'FUSION_TYPE')
        assert hasattr(Config, 'APPROACH')
        assert hasattr(Config, 'PRETRAINED')
    
    def test_config_values(self):
        """Config değerleri doğru mu?"""
        assert Config.IMG_SIZE == 384
        assert Config.NUM_CLASSES == 4
        assert Config.BATCH_SIZE > 0
        assert Config.NUM_EPOCHS > 0
        assert Config.FUSION_TYPE in ['early', 'late', 'attention']
        assert Config.APPROACH in ['bilateral', 'multi_view']
        assert Config.IN_CHANNELS == 1  # Grayscale
    
    def test_learning_rate_method(self):
        """get_learning_rate metodu çalışıyor mu?"""
        lr = Config.get_learning_rate('efficientnet_b0')
        assert lr == 1e-4
        
        lr_swin = Config.get_learning_rate('swin_tiny_patch4_window7_224')
        assert lr_swin == 1e-5
        
        # Tanımsız model için default
        lr_unknown = Config.get_learning_rate('unknown_model')
        assert lr_unknown == 1e-4
    
    def test_device(self):
        """Device doğru mu?"""
        assert isinstance(Config.DEVICE, torch.device)
    
    def test_scheduler_type(self):
        """Scheduler type doğru mu?"""
        assert Config.SCHEDULER_TYPE in ['cosine', 'plateau', 'onecycle']
    
    def test_class_weights_config(self):
        """Class imbalance config doğru mu?"""
        assert isinstance(Config.USE_CLASS_WEIGHTS, bool)
        assert isinstance(Config.USE_OVERSAMPLING, bool)
        assert Config.OVERSAMPLING_STRATEGY in ['auto', 'threshold', 'manual']
        assert 0 < Config.OVERSAMPLING_THRESHOLD <= 1
    
    def test_fast_config(self):
        """FastTestConfig doğru mu?"""
        assert FastTestConfig.NUM_EPOCHS == 2
        assert FastTestConfig.BATCH_SIZE == 4
        assert FastTestConfig.USE_GRADCAM == False
        assert FastTestConfig.USE_OVERSAMPLING == False


class TestConfigInheritance:
    """Config kalıtımı testleri"""
    
    def test_fast_config_inherits(self):
        """FastTestConfig, Config'den inherit ediyor mu?"""
        # FastTestConfig override ettiği değerler
        assert FastTestConfig.NUM_EPOCHS == 2
        
        # Config'den gelen değerler
        assert FastTestConfig.IMG_SIZE == 384
        assert FastTestConfig.NUM_CLASSES == 4
    
    def test_small_config_inherits(self):
        """SmallDataConfig override'ları doğru mu?"""
        assert SmallDataConfig.NUM_EPOCHS == 50
        assert SmallDataConfig.BATCH_SIZE == 16
        assert SmallDataConfig.OVERSAMPLING_STRATEGY == "auto"
    
    def test_large_config_inherits(self):
        """LargeDataConfig override'ları doğru mu?"""
        assert LargeDataConfig.BATCH_SIZE == 32
        assert LargeDataConfig.NUM_EPOCHS == 100
        assert LargeDataConfig.OVERSAMPLING_STRATEGY == "threshold"
        assert LargeDataConfig.PATIENCE == 15
    
    def test_swin_config(self):
        """SwinConfig özel ayarları doğru mu?"""
        assert SwinConfig.MODEL_NAME == "swin_tiny_patch4_window7_224"
        assert SwinConfig.BATCH_SIZE == 12
        assert SwinConfig.NUM_EPOCHS == 70


class TestConfigMethods:
    """Config metodları testleri"""
    
    def test_print_config_no_error(self, capsys):
        """print_config() hata vermeden çalışıyor mu?"""
        Config.print_config()
        captured = capsys.readouterr()
        assert "PROJE KONFİGÜRASYONU" in captured.out
        assert "Model:" in captured.out