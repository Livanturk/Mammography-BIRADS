"""
Pytest Fixtures - Ortak test verileri
"""

import pytest
import torch
import sys
from pathlib import Path

# src/ klasörünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config, FastTestConfig, SmallDataConfig, LargeDataConfig


@pytest.fixture
def config():
    """Varsayılan config fixture"""
    return Config


@pytest.fixture
def fast_config():
    """Hızlı test config fixture"""
    return FastTestConfig


@pytest.fixture
def small_config():
    """Küçük veri config fixture"""
    return SmallDataConfig


@pytest.fixture
def large_config():
    """Büyük veri config fixture"""
    return LargeDataConfig


@pytest.fixture
def device():
    """Device fixture (CPU/GPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_bilateral_batch():
    """Dummy batch verisi (bilateral - late/attention fusion için)"""
    return {
        'cc': torch.randn(4, 1, 384, 384),   # 4 örnek, 1 kanal (grayscale), 384x384
        'mlo': torch.randn(4, 1, 384, 384),
        'labels': torch.tensor([0, 1, 2, 3])  # 4 sınıf (BI-RADS 1, 2, 4, 5)
    }


@pytest.fixture
def dummy_early_fusion_batch():
    """Dummy batch verisi (early fusion için)"""
    return {
        'images': torch.randn(4, 2, 384, 384),  # 2 kanal (CC + MLO)
        'labels': torch.tensor([0, 1, 2, 3])
    }