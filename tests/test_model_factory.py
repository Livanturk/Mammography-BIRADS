"""
Model Factory testleri
pytest -v tests/test_model_factory.py
"""

import pytest
import torch
from models.model_factory import get_model, get_model_info, list_available_models
from config import Config


class TestModelFactory:
    """Model factory temel testleri"""
    
    def test_model_creation(self):
        """Model oluşturuluyor mu?"""
        model = get_model(Config)
        assert model is not None
    
    def test_model_type(self):
        """Model doğru tip mi?"""
        model = get_model(Config)
        assert isinstance(model, torch.nn.Module)
    
    def test_model_parameters(self):
        """Model parametreleri var mı?"""
        model = get_model(Config)
        params = list(model.parameters())
        assert len(params) > 0, "Model'de hiç parametre yok!"
    
    def test_model_trainable(self):
        """Model eğitilebilir mi?"""
        model = get_model(Config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Hiç trainable parametre yok!"


class TestModelForwardPass:
    """Forward pass testleri"""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_forward_pass_different_batch_sizes(self, batch_size):
        """Farklı batch size'larla forward pass"""
        model = get_model(Config)
        model.eval()
        
        # Bilateral için 2 kanal (CC + MLO)
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(batch_size, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (batch_size, Config.NUM_CLASSES)
        assert output.shape == expected_shape, \
            f"Output shape yanlış! Beklenen: {expected_shape}, Gelen: {output.shape}"
    
    def test_output_values_are_finite(self):
        """Output değerleri sonlu mu? (inf/nan kontrolü)"""
        model = get_model(Config)
        model.eval()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(4, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        assert torch.isfinite(output).all(), "Output'ta inf veya nan var!"
    
    def test_output_range(self):
        """Output değerleri makul aralıkta mı?"""
        model = get_model(Config)
        model.eval()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(4, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        # Logits genelde -10 ile +10 arasında
        assert output.abs().max() < 100, \
            f"Output değerleri çok büyük: {output.abs().max():.2f}"
    
    def test_gradient_flow(self):
        """Gradient akışı var mı?"""
        model = get_model(Config)
        model.train()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(4, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        target = torch.tensor([0, 1, 2, 3])
        
        # Forward
        output = model(x)
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
    
    def test_deterministic_in_eval_mode(self):
        """Eval modda deterministik mi?"""
        torch.manual_seed(42)
        model = get_model(Config)
        model.eval()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(2, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5), \
            "Eval modda aynı input farklı output veriyor!"


class TestModelArchitectures:
    """Farklı model mimarileri testleri"""
    
    @pytest.mark.slow
    def test_efficientnet_b0(self):
        """EfficientNet-B0 çalışıyor mu?"""
        # Geçici config
        class TempConfig:
            MODEL_NAME = 'efficientnet_b0'
            PRETRAINED = True
            NUM_CLASSES = 4
            APPROACH = 'bilateral'
            IMG_SIZE = 384
            IN_CHANNELS = 1
        
        model = get_model(TempConfig)
        
        x = torch.randn(2, 2, 384, 384)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 4)
    
    @pytest.mark.slow
    def test_resnet50(self):
        """ResNet50 çalışıyor mu?"""
        class TempConfig:
            MODEL_NAME = 'resnet50'
            PRETRAINED = True
            NUM_CLASSES = 4
            APPROACH = 'bilateral'
            IMG_SIZE = 384
            IN_CHANNELS = 1
        
        model = get_model(TempConfig)
        
        x = torch.randn(2, 2, 384, 384)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 4)


class TestModelInfo:
    """Model bilgi fonksiyonları testleri"""
    
    def test_get_model_info(self):
        """get_model_info çalışıyor mu?"""
        info = get_model_info('efficientnet_b0')
        
        assert info is not None
        assert 'name' in info
        assert 'total_params' in info
        assert info['available'] == True
    
    def test_get_model_info_invalid_model(self):
        """Geçersiz model ismi için hata var mı?"""
        info = get_model_info('invalid_model_name_xyz123')
        
        assert info is not None
        assert info['available'] == False
        assert 'error' in info
    
    def test_list_available_models(self):
        """list_available_models çalışıyor mu?"""
        models = list_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # EfficientNet ailesi var mı?
        efficientnets = [m for m in models if 'efficientnet' in m]
        assert len(efficientnets) > 0, "Hiç EfficientNet modeli yok!"


@pytest.mark.gpu
class TestModelGPU:
    """GPU testleri (GPU varsa)"""
    
    def test_model_to_gpu(self):
        """Model GPU'ya taşınabiliyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        model = get_model(Config)
        device = torch.device('cuda')
        model = model.to(device)
        
        # Model GPU'da mı?
        first_param_device = next(model.parameters()).device
        assert first_param_device.type == 'cuda', \
            f"Model GPU'da değil! Device: {first_param_device}"
    
    def test_forward_on_gpu(self):
        """GPU'da forward pass çalışıyor mu?"""
        if not torch.cuda.is_available():
            pytest.skip("GPU yok")
        
        device = torch.device('cuda')
        model = get_model(Config).to(device)
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(4, in_channels, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == 'cuda'
        assert output.shape == (4, Config.NUM_CLASSES)


class TestEdgeCases:
    """Edge case testleri"""
    
    def test_batch_size_one(self):
        """Batch size 1 çalışıyor mu?"""
        model = get_model(Config)
        model.eval()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(1, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, Config.NUM_CLASSES)
    
    def test_large_batch_size(self):
        """Büyük batch size çalışıyor mu?"""
        model = get_model(Config)
        model.eval()
        
        in_channels = 2 if Config.APPROACH == "bilateral" else 4
        x = torch.randn(64, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (64, Config.NUM_CLASSES)
    
    def test_model_can_be_saved_and_loaded(self):
        """Model kaydedilip yüklenebiliyor mu?"""
        import tempfile
        
        model = get_model(Config)
        
        # Kaydet
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            
            # Yeni model oluştur
            model2 = get_model(Config)
            
            # Yükle
            model2.load_state_dict(torch.load(tmp.name))
            
            # Aynı output mu?
            model.eval()
            model2.eval()
            
            in_channels = 2 if Config.APPROACH == "bilateral" else 4
            x = torch.randn(2, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
            
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)
            
            assert torch.allclose(out1, out2, rtol=1e-5), \
                "Kaydedilen ve yüklenen model farklı output veriyor!"