"""
Model karşılaştırmaları ve Grid Search ile hyperparameter ayarı
+ Grad-CAM görselleştirmeleri
"""

import torch
import torch.nn.functional as F
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from training.train import train_model, get_dataloaders
from utils.gradcam import GradCAM
from models.model_factory import get_model
from models.late_fusion_model import get_late_fusion_model
from models.attention_fusion_model import get_attention_fusion_model

class ExperimentRunner:
    """
    Experiment yönetimi ve karşılaştırmalar
    """
    def __init__(self, experiment_name="mamografi_birads_comparison"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        self.results = []
        self.gradcam_samples = Config.GRADCAM_SAMPLES  # Her experiment için kaç örnek

        # Sonuç klasörleri
        self.results_dir = Path("./experiment_results")
        self.gradcam_base_dir = Path("./gradcam_outputs")
        self.results_dir.mkdir(exist_ok=True)
        self.gradcam_base_dir.mkdir(exist_ok=True)
    
    def run_experiment(self, config_dict, run_name, generate_gradcam=True):
        """
        Tek bir experimenti çalıştırır
        Input:
            - config_dict = config overrideları
            - run_name = MLflow run ismi
            - generate_gradcam = Grad-CAM görselleştirmesi üretilsin mi?

        Output:
            - best_accuracy = en iyi val accuracy
        """

        # Config oluşturma - Config class'ını doğrudan modifiye et
        # Override'lardan önce orijinal değerleri sakla
        original_values = {}
        for key, value in config_dict.items():
            if hasattr(Config, key):
                original_values[key] = getattr(Config, key)
            setattr(Config, key, value)

        # Config referansı
        config = Config

        print(f"\n{'='*70}")
        print(f" EXPERIMENT: {run_name}")
        print('='*70)
        print(f"Config overrides:")
        for key, value in config_dict.items():
            print(f"   {key}: {value}")

        # Training
        try:
            best_acc = train_model(config, run_name=run_name)

            # Sonuç kaydetme
            result = {
                'run_name': run_name,
                'best_accuracy': best_acc,
                'timestamp': datetime.now().isoformat(),
                **config_dict
            }

            self.results.append(result)

            # Grad-CAM üret
            if generate_gradcam and config.USE_GRADCAM:
                checkpoint_path = config.CHECKPOINT_DIR / f"best_{run_name}.pth"
                if checkpoint_path.exists():
                    gradcam_dir = self.gradcam_base_dir / run_name
                    self._generate_gradcam_for_experiment(
                        checkpoint_path, config, gradcam_dir, config_dict
                    )
                else:
                    print(f"   Checkpoint bulunamadı, Grad-CAM atlanıyor: {checkpoint_path}")

            return best_acc

        except Exception as e:
            print(f"Experiment başarısız: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

        finally:
            # Orijinal config değerlerini geri yükle
            for key, value in original_values.items():
                setattr(Config, key, value)

    def _get_target_layer(self, model, fusion_type, model_name):
        """
        Model ve fusion type'a göre Grad-CAM için target layer belirle
        """
        if fusion_type == "early":
            # Early fusion - tek encoder
            if 'efficientnet' in model_name:
                return model.blocks[-1]
            elif 'resnet' in model_name:
                return model.layer4
            elif 'convnext' in model_name:
                return model.stages[-1]
            elif 'swin' in model_name:
                return model.layers[-1]
            else:
                return model.blocks[-1]
        else:
            # Late fusion veya attention - cc_encoder kullan
            if 'efficientnet' in model_name:
                return model.cc_encoder.blocks[-1]
            elif 'resnet' in model_name:
                return model.cc_encoder.layer4
            elif 'convnext' in model_name:
                return model.cc_encoder.stages[-1]
            elif 'swin' in model_name:
                return model.cc_encoder.layers[-1]
            else:
                return model.cc_encoder.blocks[-1]

    def _generate_gradcam_for_experiment(self, checkpoint_path, config, save_dir, config_dict):
        """
        Bir experiment için Grad-CAM görselleştirmelerini üret
        """
        import cv2
        from dataset.dataset import BilateralMammogramDataset, get_transforms

        print(f"\n   Grad-CAM üretiliyor: {save_dir}")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Config değerlerini geçici olarak ayarla
        fusion_type = config_dict.get('FUSION_TYPE', config.FUSION_TYPE)
        model_name = config_dict.get('MODEL_NAME', config.MODEL_NAME)

        # Model yükle
        if fusion_type == "early":
            model = get_model(config)
        elif fusion_type == "late":
            model = get_late_fusion_model(config)
        else:
            model = get_attention_fusion_model(config)

        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        model.eval()

        # Target layer
        target_layer = self._get_target_layer(model, fusion_type, model_name)

        # Grad-CAM objesi
        gradcam = GradCAM(model, target_layer)

        # Test dataseti
        val_transform = get_transforms(config, is_train=False)
        val_dataset = BilateralMammogramDataset(
            config=config,
            class_folders=config.TEST_CLASSES,
            transform=val_transform,
            is_train=False
        )

        # Rastgele örnekler seç (her class'tan eşit)
        samples_per_class = max(1, self.gradcam_samples // config.NUM_CLASSES)
        selected_indices = []

        for class_idx in range(config.NUM_CLASSES):
            class_indices = [i for i, (_, _, label) in enumerate(val_dataset.samples)
                           if label == class_idx]
            if class_indices:
                n_select = min(samples_per_class, len(class_indices))
                selected_indices.extend(random.sample(class_indices, n_select))

        # BI-RADS mapping
        birads_mapping = {0: 1, 1: 2, 2: 4, 3: 5}

        for idx in selected_indices:
            try:
                if fusion_type == "early":
                    image, label = val_dataset[idx]
                    image = image.unsqueeze(0).to(config.DEVICE)

                    # Forward
                    with torch.no_grad():
                        output = model(image)
                        pred_class = output.argmax(dim=1).item()
                        probs = F.softmax(output, dim=1)[0]

                    # Grad-CAM üret
                    cam = gradcam.generate_cam(image, target_class=pred_class)

                    # Görselleştir
                    orig_img = image[0, 0].cpu().numpy()
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
                else:
                    cc_img, mlo_img, label = val_dataset[idx]
                    cc_img = cc_img.unsqueeze(0).to(config.DEVICE)
                    mlo_img = mlo_img.unsqueeze(0).to(config.DEVICE)

                    # Forward
                    with torch.no_grad():
                        output = model(cc_img, mlo_img)
                        pred_class = output.argmax(dim=1).item()
                        probs = F.softmax(output, dim=1)[0]

                    # Grad-CAM üret (CC görüntüsü için)
                    cam = gradcam.generate_cam(cc_img, target_class=pred_class)

                    # Görselleştir
                    orig_img = cc_img[0, 0].cpu().numpy()
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

                # Heatmap ve overlay oluştur
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                original_uint8 = np.uint8(255 * orig_img)
                original_rgb = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB)

                overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)

                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                true_birads = birads_mapping[label]
                pred_birads = birads_mapping[pred_class]
                confidence = probs[pred_class].item()

                fig.suptitle(
                    f'True: BI-RADS {true_birads} | Pred: BI-RADS {pred_birads} ({confidence:.1%})',
                    fontsize=14, fontweight='bold'
                )

                axes[0].imshow(orig_img, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap')
                axes[1].axis('off')

                axes[2].imshow(overlay)
                axes[2].set_title('Overlay')
                axes[2].axis('off')

                plt.tight_layout()

                # Kaydet
                correct = "correct" if label == pred_class else "wrong"
                filename = f"sample{idx}_true{true_birads}_pred{pred_birads}_{correct}.png"
                plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"      Grad-CAM hatası (idx={idx}): {e}")
                continue

        print(f"      {len(selected_indices)} Grad-CAM görüntüsü kaydedildi")
        
    def compare_all_models(self):
        """
        Tüm modelleri karşılaştır (default parametrelerle)
        """
        models = [
            'efficientnet_b0',
            'resnet50',
            'convnext_tiny',
            'swin_tiny_patch4_window7_224',
        ]

        print("\n" + "="*70)
        print(" TÜM MODELLERİ KARŞILAŞTIRMA")
        print("="*70)
        print(f"Test edilecek modeller: {len(models)}")
        print(f"Fusion type: {Config.FUSION_TYPE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")

        for model_name in models:
            config_dict = {
                'MODEL_NAME': model_name,
                'MLFLOW_EXPERIMENT_NAME': self.experiment_name
            }

            run_name = f"{model_name}_{Config.FUSION_TYPE}_default"

            self.run_experiment(config_dict, run_name)

        # Sonuçları yazdır
        self._print_comparison_results()

    def full_comparison(self, generate_gradcam=True):
        """
        Tüm model + fusion type kombinasyonlarını karşılaştır
        Her experiment için Grad-CAM görselleştirmeleri üretir

        Args:
            generate_gradcam: Grad-CAM üretilsin mi?
        """
        models = [
            'efficientnet_b0',
            'resnet50',
            'convnext_tiny',
            'swin_tiny_patch4_window7_224',
        ]

        fusion_types = ['early', 'late', 'attention']

        total_experiments = len(models) * len(fusion_types)

        print("\n" + "="*70)
        print(" TAM KARŞILAŞTIRMA: TÜM MODEL + FUSION KOMBİNASYONLARI")
        print("="*70)
        print(f"Modeller: {models}")
        print(f"Fusion tipleri: {fusion_types}")
        print(f"Toplam experiment: {total_experiments}")
        print(f"Grad-CAM: {'Aktif' if generate_gradcam else 'Pasif'}")
        print(f"Grad-CAM örnek sayısı: {self.gradcam_samples}")
        print("="*70)

        experiment_count = 0

        for model_name in models:
            for fusion_type in fusion_types:
                experiment_count += 1

                print(f"\n[{experiment_count}/{total_experiments}]")

                config_dict = {
                    'MODEL_NAME': model_name,
                    'FUSION_TYPE': fusion_type,
                    'MLFLOW_EXPERIMENT_NAME': self.experiment_name,
                    'USE_GRADCAM': generate_gradcam,
                }

                run_name = f"{model_name}_{fusion_type}"

                self.run_experiment(config_dict, run_name, generate_gradcam=generate_gradcam)

        # Sonuçları yazdır ve kaydet
        self._print_comparison_results()

        # Karşılaştırma grafikleri oluştur
        self._create_comparison_plots()

        # Grad-CAM özet sayfası oluştur
        if generate_gradcam:
            self._create_gradcam_summary()

    def _create_comparison_plots(self):
        """
        Model ve fusion karşılaştırma grafikleri oluştur
        """
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        # Model vs Fusion heatmap
        if 'MODEL_NAME' in df.columns and 'FUSION_TYPE' in df.columns:
            pivot = df.pivot_table(
                values='best_accuracy',
                index='MODEL_NAME',
                columns='FUSION_TYPE',
                aggfunc='mean'
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGn',
                       cbar_kws={'label': 'Validation Accuracy'})
            plt.title('Model vs Fusion Type Accuracy Comparison', fontsize=14, fontweight='bold')
            plt.xlabel('Fusion Type')
            plt.ylabel('Model')
            plt.tight_layout()

            plot_path = self.results_dir / f"comparison_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n Karşılaştırma heatmap kaydedildi: {plot_path}")
            plt.close()

            # Bar plot - Model bazlı
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df, x='MODEL_NAME', y='best_accuracy', hue='FUSION_TYPE')
            plt.title('Model Performance by Fusion Type', fontsize=14, fontweight='bold')
            plt.xlabel('Model')
            plt.ylabel('Validation Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Fusion Type')

            # Değerleri barların üzerine yaz
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)

            plt.tight_layout()

            bar_path = self.results_dir / f"comparison_barplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(bar_path, dpi=150, bbox_inches='tight')
            print(f" Karşılaştırma barplot kaydedildi: {bar_path}")
            plt.close()

    def _create_gradcam_summary(self):
        """
        Tüm Grad-CAM görsellerinin özet sayfasını oluştur
        """
        print("\n Grad-CAM özet sayfası oluşturuluyor...")

        # Her experiment için bir örnek seç
        summary_images = []

        for result in self.results:
            run_name = result['run_name']
            gradcam_dir = self.gradcam_base_dir / run_name

            if gradcam_dir.exists():
                images = list(gradcam_dir.glob("*.png"))
                if images:
                    # İlk doğru tahmini bul, yoksa ilkini al
                    correct_images = [img for img in images if "correct" in img.name]
                    sample_img = correct_images[0] if correct_images else images[0]
                    summary_images.append({
                        'run_name': run_name,
                        'path': sample_img,
                        'accuracy': result['best_accuracy']
                    })

        if not summary_images:
            print("   Özet için Grad-CAM görüntüsü bulunamadı")
            return

        # Özet grid oluştur
        n_images = len(summary_images)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, img_info in enumerate(summary_images):
            row = idx // n_cols
            col = idx % n_cols

            img = plt.imread(img_info['path'])
            axes[row, col].imshow(img)
            axes[row, col].set_title(
                f"{img_info['run_name']}\nAcc: {img_info['accuracy']:.4f}",
                fontsize=10
            )
            axes[row, col].axis('off')

        # Boş eksenleri gizle
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Grad-CAM Summary - All Experiments', fontsize=16, fontweight='bold')
        plt.tight_layout()

        summary_path = self.results_dir / f"gradcam_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f" Grad-CAM özet kaydedildi: {summary_path}")
        plt.close()
    
    def grid_search_hyperparameters(self, model_name='efficientnet_b0'):
        """
        Bir model için hiperparametre grid search
        """
        print("\n" + "="*70)
        print(f"GRID SEARCH: {model_name}")
        print("="*70)
        
        # Grid parametreleri
        grid = {
            'batch_size': [8, 16, 32],
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'scheduler': ['cosine', 'plateau'],
            'label_smoothing': [0.0, 0.1],
        }
        
        print("\nGrid parametreleri:")
        for key, values in grid.items():
            print(f"   {key}: {values}")
        
        # Toplam kombinasyon sayısı
        total_combinations = 1
        for values in grid.values():
            total_combinations *= len(values)
        
        print(f"\nToplam kombinasyon: {total_combinations}")
        
        # Grid search
        experiment_count = 0
        
        for batch_size in grid['batch_size']:
            for lr in grid['learning_rate']:
                for scheduler in grid['scheduler']:
                    for label_smoothing in grid['label_smoothing']:
                        
                        experiment_count += 1
                        
                        config_dict = {
                            'MODEL_NAME': model_name,
                            'BATCH_SIZE': batch_size,
                            'LEARNING_RATE': lr,
                            'SCHEDULER_TYPE': scheduler,
                            'LABEL_SMOOTHING_FACTOR': label_smoothing,
                            'USE_LABEL_SMOOTHING': label_smoothing > 0,
                            'MLFLOW_EXPERIMENT_NAME': self.experiment_name
                        }
                        
                        run_name = (
                            f"{model_name}_bs{batch_size}_lr{lr:.0e}_"
                            f"{scheduler}_ls{label_smoothing}"
                        )
                        
                        print(f"\n[{experiment_count}/{total_combinations}] {run_name}")
                        
                        self.run_experiment(config_dict, run_name)
        
        # Sonuçları analiz et
        self._analyze_grid_search_results(model_name)
    
    def compare_fusion_types(self, model_name='efficientnet_b0'):
        """
        Farklı fusion tiplerini karşılaştır
        
        Args:
            model_name: Hangi model ile test edilsin?
        """
        fusion_types = ['early', 'late', 'attention']
        
        print("\n" + "="*70)
        print(f" Fusion tiplerinin karşılaştırması: {model_name}")
        print("="*70)
        
        for fusion_type in fusion_types:
            config_dict = {
                'MODEL_NAME': model_name,
                'FUSION_TYPE': fusion_type,
                'MLFLOW_EXPERIMENT_NAME': self.experiment_name
            }
            
            run_name = f"{model_name}_{fusion_type}_fusion"
            
            self.run_experiment(config_dict, run_name)
        
        self._print_comparison_results()
    
    def compare_oversampling_strategies(self, model_name='efficientnet_b0'):
        """
        Farklı oversampling stratejilerini karşılaştır
        """
        strategies = [
            ('none', False, None),
            ('auto', True, 'auto'),
            ('threshold', True, 'threshold'),
        ]
        
        print("\n" + "="*70)
        print(f" Oversampling strateji karşılaştırması: {model_name}")
        print("="*70)
        
        for name, use_oversampling, strategy in strategies:
            config_dict = {
                'MODEL_NAME': model_name,
                'USE_OVERSAMPLING': use_oversampling,
                'OVERSAMPLING_STRATEGY': strategy if use_oversampling else None,
                'MLFLOW_EXPERIMENT_NAME': self.experiment_name
            }
            
            run_name = f"{model_name}_oversampling_{name}"
            
            self.run_experiment(config_dict, run_name)
        
        self._print_comparison_results()
    
    def custom_experiments(self):
        """
        Özel experiment'ler (istediğin kombinasyonları test et,  ben bunları ekledim kendimce)
        """
        experiments = [
            # Experiment 1: default
            {
                'name': 'best_guess_efficientnet',
                'config': {
                    'MODEL_NAME': 'efficientnet_b0',
                    'FUSION_TYPE': 'attention',
                    'BATCH_SIZE': 16,
                    'NUM_EPOCHS': 50,
                    'LEARNING_RATE': 1e-4,
                    'SCHEDULER_TYPE': 'cosine',
                    'USE_LABEL_SMOOTHING': True,
                    'LABEL_SMOOTHING_FACTOR': 0.1,
                    'USE_OVERSAMPLING': True,
                    'OVERSAMPLING_STRATEGY': 'auto',
                }
            },
            
            # Experiment 2: Swin transformer optimized
            {
                'name': 'optimized_swin',
                'config': {
                    'MODEL_NAME': 'swin_tiny_patch4_window7_224',
                    'FUSION_TYPE': 'attention',
                    'BATCH_SIZE': 12,
                    'NUM_EPOCHS': 70,
                    'LEARNING_RATE': 1e-5,
                    'SCHEDULER_TYPE': 'cosine',
                    'USE_LABEL_SMOOTHING': True,
                }
            },
            
            # Experiment 3: ResNet aggressive
            {
                'name': 'resnet_aggressive',
                'config': {
                    'MODEL_NAME': 'resnet50',
                    'FUSION_TYPE': 'attention',
                    'BATCH_SIZE': 32,
                    'NUM_EPOCHS': 100,
                    'LEARNING_RATE': 1e-4,
                    'SCHEDULER_TYPE': 'onecycle',
                }
            },
        ]
        
        print("\n" + "="*70)
        print(" CUSTOM EXPERIMENTS")
        print("="*70)
        print(f"Toplam experiment: {len(experiments)}")
        
        for exp in experiments:
            config_dict = exp['config']
            config_dict['MLFLOW_EXPERIMENT_NAME'] = self.experiment_name
            
            self.run_experiment(config_dict, exp['name'])
        
        self._print_comparison_results()
    
    def _print_comparison_results(self):
        """
        Sonuçları tablo halinde yazdır
        """
        if not self.results:
            print("\n Henüz sonuç yok!")
            return
        
        # DataFrame oluştur
        df = pd.DataFrame(self.results)
        
        # Accuracy'ye göre sırala
        df = df.sort_values('best_accuracy', ascending=False)
        
        print("\n" + "="*70)
        print(" SONUÇLAR (Accuracy'ye göre sıralı)")
        print("="*70)
        
        # Tablo yazdır
        print(df.to_string(index=False))
        
        # En iyi model
        best = df.iloc[0]
        print(f"\n EBest model:")
        print(f"   Run: {best['run_name']}")
        print(f"   Accuracy: {best['best_accuracy']:.4f}")
        
        # Sonuçları CSV'ye kaydet
        results_dir = Path("./experiment_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = results_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n Sonuçlar kaydedildi: {csv_path}")
    
    def _analyze_grid_search_results(self, model_name):
        """
        Grid search sonuçlarını analiz et ve görselleştir
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Model filtrele
        df_model = df[df['MODEL_NAME'] == model_name]
        
        if df_model.empty:
            return
        
        print("\n" + "="*70)
        print(f" GRID SEARCH ANALİZİ: {model_name}")
        print("="*70)
        
        # Parametrelere göre groupby ve ortalama accuracy
        if 'BATCH_SIZE' in df_model.columns:
            print("\n Batch Size Etkisi:")
            batch_effect = df_model.groupby('BATCH_SIZE')['best_accuracy'].agg(['mean', 'std', 'count'])
            print(batch_effect)
        
        if 'LEARNING_RATE' in df_model.columns:
            print("\n Learning Rate Etkisi:")
            lr_effect = df_model.groupby('LEARNING_RATE')['best_accuracy'].agg(['mean', 'std', 'count'])
            print(lr_effect)
        
        if 'SCHEDULER_TYPE' in df_model.columns:
            print("\n Scheduler Etkisi:")
            scheduler_effect = df_model.groupby('SCHEDULER_TYPE')['best_accuracy'].agg(['mean', 'std', 'count'])
            print(scheduler_effect)
        
        if 'LABEL_SMOOTHING_FACTOR' in df_model.columns:
            print("\n Label Smoothing Etkisi:")
            ls_effect = df_model.groupby('LABEL_SMOOTHING_FACTOR')['best_accuracy'].agg(['mean', 'std', 'count'])
            print(ls_effect)
        
        # Heatmap oluştur (batch_size vs learning_rate)
        if 'BATCH_SIZE' in df_model.columns and 'LEARNING_RATE' in df_model.columns:
            self._plot_heatmap(df_model, model_name)
    
    def _plot_heatmap(self, df, model_name):
        """
        Batch size vs Learning rate heatmap
        """
        pivot = df.pivot_table(
            values='best_accuracy',
            index='BATCH_SIZE',
            columns='LEARNING_RATE',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title(f'{model_name}: Batch Size vs Learning Rate')
        plt.xlabel('Learning Rate')
        plt.ylabel('Batch Size')
        
        # Kaydet
        results_dir = Path("./experiment_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = results_dir / f"heatmap_{model_name}_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n Heatmap kaydedildi: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Model Karşılaştırma')

    parser.add_argument('--mode', type=str, required=True,
                       choices=['all_models', 'grid_search', 'fusion_types',
                               'oversampling', 'custom', 'full_comparison'],
                       help='Hangi karşılaştırma yapılsın?')

    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       help='Model ismi (grid_search ve diğerleri için)')

    parser.add_argument('--experiment-name', type=str,
                       default='mamografi_birads_comparison',
                       help='MLflow experiment ismi')

    parser.add_argument('--no-gradcam', action='store_true',
                       help='Grad-CAM görselleştirmelerini devre dışı bırak')

    parser.add_argument('--gradcam-samples', type=int, default=10,
                       help='Her experiment için kaç Grad-CAM örneği üretilsin?')

    args = parser.parse_args()

    # Experiment runner
    runner = ExperimentRunner(experiment_name=args.experiment_name)

    # Grad-CAM örnek sayısını ayarla
    runner.gradcam_samples = args.gradcam_samples

    print("\n" + "="*70)
    print("Model karşılaştırma")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"MLflow Experiment: {args.experiment_name}")
    print(f"Grad-CAM: {'Pasif' if args.no_gradcam else 'Aktif'}")

    # Mode'a göre çalıştır
    if args.mode == 'all_models':
        runner.compare_all_models()

    elif args.mode == 'grid_search':
        runner.grid_search_hyperparameters(model_name=args.model)

    elif args.mode == 'fusion_types':
        runner.compare_fusion_types(model_name=args.model)

    elif args.mode == 'oversampling':
        runner.compare_oversampling_strategies(model_name=args.model)

    elif args.mode == 'custom':
        runner.custom_experiments()

    elif args.mode == 'full_comparison':
        runner.full_comparison(generate_gradcam=not args.no_gradcam)

    print("\n" + "="*70)
    print("Karşılaştırma tamam")
    print("="*70)
    print("\nMLflow UI'ı başlatmak için:")
    print(f"   mlflow ui --backend-store-uri {Config.MLFLOW_TRACKING_URI}")
    print("\nhttp://localhost:5000")

    # Sonuç klasörlerini göster
    print(f"\nSonuçlar:")
    print(f"   Experiment results: {runner.results_dir}")
    print(f"   Grad-CAM outputs: {runner.gradcam_base_dir}")


if __name__ == "__main__":
    main()
        