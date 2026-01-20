"""
Grad-CAM Implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class GradCAM:
    """
    Model hangi piksellere bakarak karar veriyor?
    """
    
    def __init__(self, model, target_layer):
        """
        model = PyTorch modeli veriyoruz
        target_layer = hangi katmanı görselleştireceğiz? (örn: model.cc_encoder.blocks[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hookları kaydet
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Forward pass'te activation'ı kaydet"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Backward pass'te gradient'i kaydet"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Grad-CAM haritası üret
        
        Args:
            input_image: (1, C, H, W) tensor
            target_class: Hangi sınıf için? (None ise en yüksek)
        
        Returns:
            cam: (H, W) heatmap
        """
        self.model.eval()
        
        # Forward pass
        if hasattr(self.model, 'cc_encoder'):
            # Late fusion veya attention fusion
            output = self.model(input_image, input_image)
        else:
            # Early fusion
            output = self.model(input_image)
        
        # Target class'ı belirle
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Target class'ın score'unu al
        score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Gradients ve activations
        gradients = self.gradients[0]  # (C, H', W')
        activations = self.activations[0]  # (C, H', W')
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU (negatif değerleri sıfırla)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        
        return cam
    
    def visualize(self, input_image, target_class=None, original_img=None, save_path=None):
        """
        Grad-CAM'i görselleştir
        
        Args:
            input_image: (1, C, H, W) tensor (normalized)
            target_class: Hedef sınıf
            original_img: (H, W) numpy array (orjinal görüntü)
            save_path: Kaydedilecek yer
        """
        # CAM üret
        cam = self.generate_cam(input_image, target_class)
        
        # Eğer orjinal görüntü verilmemişse input'tan al
        if original_img is None:
            img = input_image[0, 0].cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            original_img = img
        
        # Heatmap oluştur
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        original_uint8 = np.uint8(255 * original_img)
        original_rgb = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Kaydedildi: {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_model_attention(model, image_path, config, save_dir=None):
    """
    Bir görüntü için Grad-CAM görselleştir
    
    Args:
        model: Eğitilmiş model
        image_path: Görüntü yolu
        config: Config objesi
        save_dir: Kaydedilecek klasör
    """
    from PIL import Image
    from dataset.dataset import get_transforms
    
    # Görüntüyü yükle
    img = Image.open(image_path).convert('L')
    img_array = np.array(img) / 255.0
    
    # Transform
    transform = get_transforms(config, is_train=False)
    img_tensor = transform(image=np.array(img))['image'].unsqueeze(0)
    
    # Device'a taşı
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Target layer belirle
    if hasattr(model, 'cc_encoder'):
        # Late fusion veya attention
        if 'efficientnet' in config.MODEL_NAME:
            target_layer = model.cc_encoder.blocks[-1]
        elif 'resnet' in config.MODEL_NAME:
            target_layer = model.cc_encoder.layer4
        elif 'convnext' in config.MODEL_NAME:
            target_layer = model.cc_encoder.stages[-1]
        elif 'swin' in config.MODEL_NAME:
            target_layer = model.cc_encoder.layers[-1]
        elif 'vit' in config.MODEL_NAME:
            target_layer = model.cc_encoder.blocks[-1]
        else:
            target_layer = model.cc_encoder.blocks[-1]
    else:
        # Early fusion
        if 'efficientnet' in config.MODEL_NAME:
            target_layer = model.blocks[-1]
        else:
            target_layer = model.layer4
    
    # Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Tahmin yap
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'cc_encoder'):
            output = model(img_tensor, img_tensor)
        else:
            output = model(img_tensor)
        
        predicted_class = output.argmax(dim=1).item()
        probabilities = F.softmax(output, dim=1)[0]
    
    # BI-RADS mapping
    birads_mapping = {0: 1, 1: 2, 2: 4, 3: 5}
    birads = birads_mapping[predicted_class]
    confidence = probabilities[predicted_class].item()
    
    print(f"\n📊 Tahmin:")
    print(f"   BI-RADS: {birads} (class {predicted_class})")
    print(f"   Confidence: {confidence:.2%}")
    
    # Save path
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(image_path).stem
        save_path = save_dir / f"gradcam_{img_name}_birads{birads}.png"
    else:
        save_path = None
    
    # Grad-CAM oluştur
    grad_cam.visualize(
        img_tensor, 
        target_class=predicted_class,
        original_img=img_array,
        save_path=save_path
    )
    
    return predicted_class, confidence


# Test kodu
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRAD-CAM TEST")
    print("="*60)
    
    print("\n Bu test için eğitilmiş bir model gerekli!")
    print("   Kullanım örneği:")
    print("""
    from models.attention_fusion_model import get_attention_fusion_model
    from utils.gradcam import visualize_model_attention
    from config import Config
    
    # Model yükle
    model = get_attention_fusion_model(Config)
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Grad-CAM
    visualize_model_attention(
        model, 
        image_path='path/to/LCC.png',
        config=Config,
        save_dir='gradcam_outputs'
    )
    """)