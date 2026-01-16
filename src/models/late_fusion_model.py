"""
Late Fusion modeli => Her görüntü (LCC-RCC-RMLO-LMLO) ayrı CNN'den geçer, sonra birleştirir (Feature Concat)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path

# Local weightlerin çekildiği klasör
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "pretrained_weights"

class LateFusionBiRADS(nn.Module):
    """
    Late Fusion Model - BI-RADS Sınıflandırma
    
    Yapı:
        CC  → CNN (encoder) → Features (1280)
                                    ↓
                                  Concat (2560)
                                    ↓
        MLO → CNN (encoder) → Features (1280)
                                    ↓
                               Fusion Layer
                                    ↓
                            BI-RADS Prediction (4)
    """
    
    def __init__(self, config):
        # config.py dosyasındaki Config classından inherit alırız
        super().__init__()
        
        self.config = config
        self.backbone = config.MODEL_NAME
        self.num_classes = config.NUM_CLASSES
        
        # CC VE MLO için ayrı encoder
        self.cc_encoder = self._create_encoder()
        self.mlo_encoder = self._create_encoder()
        
        # Feature Dimension
        feature_dim = self._get_feature_dim(self.backbone)
        
        # Fusion Classifier
        droput = 0.3
        
        