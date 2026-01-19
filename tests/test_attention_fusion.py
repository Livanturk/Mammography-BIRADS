"""
Attention Fusion Model testleri
pytest -v tests/test_attention_fusion.py
"""

import pytest
import torch
from models.attention_fusion_model import (
    get_attention_fusion_model, 
    AttentionLateFusionBiRADS,
    AttentionFusion
)
from config import Config


class TestAttentionFusionModel:
    """Attention fusion model temel testleri"""
    
    def test_model_creation(self):
        """Model oluşturuluyor mu?"""
        model = get_attention_fusion_model(Config)
        assert model is not None
        assert isinstance(model, AttentionLateFusionBiRADS)
    
    def test_model_has_components(self):
        """Model gerekli component'lere sahip mi?"""
        model = get_attention_fusion_model(Config)
        
        assert hasattr(model, 'cc_encoder')
        assert hasattr(model, 'mlo_encoder')
        assert hasattr(model, 'attention_fusion')
        assert hasattr(model, 'classifier')
    
    def test_attention_fusion_component(self):
        """Attention fusion component doğru tip mi?"""
        model = get_attention_fusion_model(Config)
        
        assert isinstance(model.attention_fusion, AttentionFusion)


class TestAttentionFusionForward:
    """Attention fusion forward pass testleri"""
    
    def test_forward_pass_without_attention(self, dummy_bilateral_batch):
        """Forward pass (attention dönmesin)"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        cc_img = dummy_bilateral_batch['cc']
        mlo_img = dummy_bilateral_batch['mlo']
        
        with torch.no_grad():
            output = model(cc_img, mlo_img, return_attention=False)
        
        expected_shape = (4, Config.NUM_CLASSES)
        assert output.shape == expected_shape, \
            f"Output shape yanlış! Beklenen: {expected_shape}, Gelen: {output.shape}"
    
    def test_forward_pass_with_attention(self, dummy_bilateral_batch):
        """Forward pass (attention dönsün)"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        cc_img = dummy_bilateral_batch['cc']
        mlo_img = dummy_bilateral_batch['mlo']
        
        with torch.no_grad():
            output, attention_weights = model(cc_img, mlo_img, return_attention=True)
        
        # Output kontrolü
        assert output.shape == (4, Config.NUM_CLASSES)
        
        # Attention weights kontrolü
        assert attention_weights.shape == (4, 2), \
            f"Attention weights shape yanlış! Beklenen: (4, 2), Gelen: {attention_weights.shape}"
    
    def test_attention_weights_sum_to_one(self, dummy_bilateral_batch):
        """Attention weights toplamı 1 mi? (Softmax)"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        with torch.no_grad():
            _, attention_weights = model(
                dummy_bilateral_batch['cc'],
                dummy_bilateral_batch['mlo'],
                return_attention=True
            )
        
        # Her örnek için attention weights toplamı 1 olmalı (softmax)
        weights_sum = attention_weights.sum(dim=1)
        
        assert torch.allclose(weights_sum, torch.ones(4), atol=1e-6), \
            f"Attention weights toplamı 1 değil! {weights_sum}"
    
    def test_attention_weights_range(self, dummy_bilateral_batch):
        """Attention weights 0-1 aralığında mı?"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        with torch.no_grad():
            _, attention_weights = model(
                dummy_bilateral_batch['cc'],
                dummy_bilateral_batch['mlo'],
                return_attention=True
            )
        
        # Tüm değerler 0-1 arasında olmalı
        assert (attention_weights >= 0).all(), "Negatif attention weight var!"
        assert (attention_weights <= 1).all(), "1'den büyük attention weight var!"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_forward_different_batch_sizes(self, batch_size):
        """Farklı batch size'larda çalışıyor mu?"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        cc_img = torch.randn(batch_size, 1, 384, 384)
        mlo_img = torch.randn(batch_size, 1, 384, 384)
        
        with torch.no_grad():
            output, attention = model(cc_img, mlo_img, return_attention=True)
        
        assert output.shape == (batch_size, Config.NUM_CLASSES)
        assert attention.shape == (batch_size, 2)
    
    def test_gradient_flow(self, dummy_bilateral_batch):
        """Gradient akışı var mı?"""
        model = get_attention_fusion_model(Config)
        model.train()
        
        cc_img = dummy_bilateral_batch['cc']
        mlo_img = dummy_bilateral_batch['mlo']
        target = dummy_bilateral_batch['labels']
        
        # Forward
        output = model(cc_img, mlo_img)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward
        loss.backward()
        
        # Attention module'e de gradient gitmeli
        attention_has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.attention_fusion.parameters()
        )
        
        assert attention_has_grads, "Attention module'e gradient gitmiyor!"


class TestAttentionWeightsMethod:
    """get_attention_weights metodu testleri"""
    
    def test_get_attention_weights(self, dummy_bilateral_batch):
        """get_attention_weights metodu çalışıyor mu?"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        weights = model.get_attention_weights(
            dummy_bilateral_batch['cc'],
            dummy_bilateral_batch['mlo']
        )
        
        assert weights.shape == (4, 2)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)
    
    def test_attention_weights_interpretability(self, dummy_bilateral_batch):
        """Attention weights yorumlanabilir mi?"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        weights = model.get_attention_weights(
            dummy_bilateral_batch['cc'],
            dummy_bilateral_batch['mlo']
        )
        
        # CC ve MLO weights
        cc_weights = weights[:, 0]
        mlo_weights = weights[:, 1]
        
        # Her ikisi de pozitif
        assert (cc_weights >= 0).all()
        assert (mlo_weights >= 0).all()
        
        # Toplamları 1
        assert torch.allclose(cc_weights + mlo_weights, torch.ones(4), atol=1e-6)


class TestAttentionMechanism:
    """Attention mekanizması testleri"""
    
    def test_different_inputs_different_attention(self):
        """Farklı input'lar farklı attention weights versin"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        # İki farklı input
        torch.manual_seed(42)
        cc_img1 = torch.randn(2, 1, 384, 384)
        mlo_img1 = torch.randn(2, 1, 384, 384)
        
        torch.manual_seed(99)
        cc_img2 = torch.randn(2, 1, 384, 384)
        mlo_img2 = torch.randn(2, 1, 384, 384)
        
        with torch.no_grad():
            weights1 = model.get_attention_weights(cc_img1, mlo_img1)
            weights2 = model.get_attention_weights(cc_img2, mlo_img2)
        
        # Farklı input'lar farklı attention vermeli
        assert not torch.allclose(weights1, weights2, rtol=1e-3), \
            "Farklı input'lar aynı attention weights veriyor!"
    
    def test_attention_affects_output(self):
        """Attention mechanism output'u etkiliyor mu?"""
        model = get_attention_fusion_model(Config)
        model.eval()
        
        cc_img = torch.randn(2, 1, 384, 384)
        mlo_img = torch.randn(2, 1, 384, 384)
        
        with torch.no_grad():
            # Normal forward
            output1 = model(cc_img, mlo_img)
            
            # Attention'ı manuel olarak değiştir
            # (Bu test için attention module'ü bypass edip manuel weight verelim)
            cc_features = model.cc_encoder(cc_img)
            mlo_features = model.mlo_encoder(mlo_img)
            
            # Manuel attention: Sadece CC kullan
            attended_manual = cc_features  # MLO'yu yok say
            output2 = model.classifier(attended_manual)
        
        # İki output farklı olmalı (attention etki ediyor)
        assert not torch.allclose(output1, output2, rtol=1e-3), \
            "Attention mechanism output'u etkilemiyor!"


@pytest.mark.gpu
class TestAttentionFusionGPU:
    """Attention fusion GPU testleri"""
    
    def test_model_to_gpu(self):
        """Model GPU'ya taşınabiliyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        model = get_attention_fusion_model(Config)
        device = torch.device('cuda')
        model = model.to(device)
        
        assert next(model.parameters()).device.type == 'cuda'
    
    def test_forward_on_gpu_with_attention(self):
        """GPU'da attention weights döndürebiliyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        device = torch.device('cuda')
        model = get_attention_fusion_model(Config).to(device)
        
        cc_img = torch.randn(4, 1, 384, 384).to(device)
        mlo_img = torch.randn(4, 1, 384, 384).to(device)
        
        with torch.no_grad():
            output, attention = model(cc_img, mlo_img, return_attention=True)
        
        assert output.device.type == 'cuda'
        assert attention.device.type == 'cuda'
        assert output.shape == (4, Config.NUM_CLASSES)
        assert attention.shape == (4, 2)