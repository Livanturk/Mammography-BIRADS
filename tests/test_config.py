"""
Config testi
pytest -v tests/test_config.py
"""

import pytest
import torch
from config import Config, FastTestConfig, SmallDataConfig, LargeDataConfig


class TestConfig:
    """Config sınıfı testleri"""
    
    def test_config_attributes(self, config):
        """Config'in gerekli attribute'ları var mı?"""
        assert hasattr(config, 'IMG_SIZE')
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(config, 'NUM_EPOCHS')
        assert hasattr(config, 'MODEL_NAME')
        assert hasattr(config, 'NUM_CLASSES')
        assert hasattr(config, 'FUSION_TYPE')
    
    def test_config_values(self, config):
        """Config değerleri doğru mu?"""
        assert config.IMG_SIZE == 384
        assert config.NUM_CLASSES == 4
        assert config.BATCH_SIZE > 0
        assert config.NUM_EPOCHS > 0
        assert config.FUSION_TYPE in ['early', 'late', 'attention']
    
    def test_learning_rate_method(self, config):
        """get_learning_rate metodu çalışıyor mu?"""
        lr = config.get_learning_rate('efficientnet_b0')
        assert lr == 1e-4
        
        lr_swin = config.get_learning_rate('swin_tiny_patch4_window7_224')
        assert lr_swin == 1e-5
        
        # Tanımsız model için default
        lr_unknown = config.get_learning_rate('unknown_model')
        assert lr_unknown == 1e-4
    
    def test_device(self, config):
        """Device doğru mu?"""
        assert isinstance(config.DEVICE, torch.device)
    
    def test_fast_config(self):
        """FastTestConfig doğru mu?"""
        assert FastTestConfig.NUM_EPOCHS == 2
        assert FastTestConfig.BATCH_SIZE == 4
        assert FastTestConfig.USE_GRADCAM == False


class TestConfigInheritance:
    """Config kalıtımı testleri"""
    
    def test_fast_config_inherits(self):
        """FastTestConfig, Config'den inherit ediyor mu?"""
        # FastTestConfig override ettiği değerler
        assert FastTestConfig.NUM_EPOCHS == 2
        
        # Config'den gelen değerler
        assert FastTestConfig.IMG_SIZE == 384
        assert FastTestConfig.NUM_CLASSES == 4
    
    def test_large_config_inherits(self):
        """LargeDataConfig override'ları doğru mu?"""
        assert LargeDataConfig.BATCH_SIZE == 32
        assert LargeDataConfig.NUM_EPOCHS == 100
        assert LargeDataConfig.OVERSAMPLING_STRATEGY == "threshold"