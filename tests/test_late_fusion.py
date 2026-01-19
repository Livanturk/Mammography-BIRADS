"""
Late Fusion Model testleri
pytest -v tests/test_late_fusion.py
"""

import pytest
import torch
from models.late_fusion_model import get_late_fusion_model, LateFusionBiRADS
from config import Config


class TestLateFusionModel:
    """Late fusion model temel testleri"""
    
    def test_model_creation(self):
        """Model oluşturuluyor mu?"""
        model = get_late_fusion_model(Config)
        assert model is not None
        assert isinstance(model, LateFusionBiRADS)
    
    def test_model_has_encoders(self):
        """Model CC ve MLO encoder'lara sahip mi?"""
        model = get_late_fusion_model(Config)
        
        assert hasattr(model, 'cc_encoder')
        assert hasattr(model, 'mlo_encoder')
        assert hasattr(model, 'fusion')
    
    def test_model_parameters(self):
        """Model parametreleri var mı?"""
        model = get_late_fusion_model(Config)
        params = list(model.parameters())
        assert len(params) > 0


class TestLateFusionForward:
    """Late fusion forward pass testleri"""
    
    def test_forward_pass(self, dummy_bilateral_batch):
        """Forward pass çalışıyor mu?"""
        model = get_late_fusion_model(Config)
        model.eval()
        
        cc_img = dummy_bilateral_batch['cc']
        mlo_img = dummy_bilateral_batch['mlo']
        
        with torch.no_grad():
            output = model(cc_img, mlo_img)
        
        expected_shape = (4, Config.NUM_CLASSES)
        assert output.shape == expected_shape, \
            f"Output shape yanlış! Beklenen: {expected_shape}, Gelen: {output.shape}"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_forward_different_batch_sizes(self, batch_size):
        """Farklı batch size'larda çalışıyor mu?"""
        model = get_late_fusion_model(Config)
        model.eval()
        
        cc_img = torch.randn(batch_size, 1, 384, 384)
        mlo_img = torch.randn(batch_size, 1, 384, 384)
        
        with torch.no_grad():
            output = model(cc_img, mlo_img)
        
        assert output.shape == (batch_size, Config.NUM_CLASSES)
    
    def test_output_is_finite(self, dummy_bilateral_batch):
        """Output sonlu mu? (inf/nan yok mu?)"""
        model = get_late_fusion_model(Config)
        model.eval()
        
        with torch.no_grad():
            output = model(
                dummy_bilateral_batch['cc'],
                dummy_bilateral_batch['mlo']
            )
        
        assert torch.isfinite(output).all(), "Output'ta inf veya nan var!"
    
    def test_gradient_flow(self, dummy_bilateral_batch):
        """Gradient akışı var mı?"""
        model = get_late_fusion_model(Config)
        model.train()
        
        cc_img = dummy_bilateral_batch['cc']
        mlo_img = dummy_bilateral_batch['mlo']
        target = dummy_bilateral_batch['labels']
        
        # Forward
        output = model(cc_img, mlo_img)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward
        loss.backward()
        
        # Gradientler var mı?
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() 
            if p.requires_grad
        )
        
        assert has_gradients, "Hiç gradient oluşmadı!"
    
    def test_both_encoders_used(self, dummy_bilateral_batch):
        """Her iki encoder da kullanılıyor mu?"""
        model = get_late_fusion_model(Config)
        model.eval()
        
        # Farklı input'lar
        cc_img1 = torch.randn(2, 1, 384, 384)
        mlo_img1 = torch.randn(2, 1, 384, 384)
        
        cc_img2 = torch.randn(2, 1, 384, 384)
        mlo_img2 = mlo_img1  # MLO aynı
        
        with torch.no_grad():
            output1 = model(cc_img1, mlo_img1)
            output2 = model(cc_img2, mlo_img2)
        
        # CC farklı ama MLO aynı → output farklı olmalı (CC encoder çalışıyor)
        assert not torch.allclose(output1, output2, rtol=1e-3), \
            "CC encoder kullanılmıyor gibi görünüyor!"


class TestLateFusionArchitecture:
    """Late fusion mimari testleri"""
    
    def test_feature_dimensions(self):
        """Feature dimension'lar doğru mu?"""
        model = get_late_fusion_model(Config)
        
        # Feature dimension'ı kontrol et
        if 'efficientnet_b0' in Config.MODEL_NAME:
            expected_dim = 1280
        elif 'resnet50' in Config.MODEL_NAME:
            expected_dim = 2048
        else:
            pytest.skip(f"Model için feature dim bilinmiyor: {Config.MODEL_NAME}")
        
        # Test forward pass ile kontrol
        model.eval()
        cc_img = torch.randn(1, 1, 384, 384)
        mlo_img = torch.randn(1, 1, 384, 384)
        
        # Encoder output'u
        with torch.no_grad():
            cc_features = model.cc_encoder(cc_img)
            mlo_features = model.mlo_encoder(mlo_img)
        
        assert cc_features.shape[1] == expected_dim, \
            f"CC feature dim yanlış! Beklenen: {expected_dim}, Gelen: {cc_features.shape[1]}"
        assert mlo_features.shape[1] == expected_dim, \
            f"MLO feature dim yanlış! Beklenen: {expected_dim}, Gelen: {mlo_features.shape[1]}"


@pytest.mark.gpu
class TestLateFusionGPU:
    """Late fusion GPU testleri"""
    
    def test_model_to_gpu(self):
        """Model GPU'ya taşınabiliyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        model = get_late_fusion_model(Config)
        device = torch.device('cuda')
        model = model.to(device)
        
        # Model GPU'da mı?
        assert next(model.parameters()).device.type == 'cuda'
    
    def test_forward_on_gpu(self):
        """GPU'da forward pass çalışıyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        device = torch.device('cuda')
        model = get_late_fusion_model(Config).to(device)
        
        cc_img = torch.randn(4, 1, 384, 384).to(device)
        mlo_img = torch.randn(4, 1, 384, 384).to(device)
        
        with torch.no_grad():
            output = model(cc_img, mlo_img)
        
        assert output.device.type == 'cuda'
        assert output.shape == (4, Config.NUM_CLASSES)