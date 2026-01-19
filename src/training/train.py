"""
Model train scripti (mlflow ile)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig
from dataset.dataset import BilateralMammogramDataset, get_transforms
from models.model_factory import get_model
from models.late_fusion_model import get_late_fusion_model
from models.attention_fusion_model import get_attention_fusion_model

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def get_config_from_args():
    """
    Terminal  kullanarak argümanları parse eder ve gerekli konfigürasyonları oluşturur
    """
    parser = argparse.ArgumentParser(description = 'Mamografi BI-RADS training')
    
    # Config seçimi
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast_test', 'small_data', 'large_data', 'swin'],
                       help='Hangi config kullanılsın?')
    
    # Parametre override'ı
    parser.add_argument('--model', type=str, help='Model ismi (efficientnet_b0, resnet50, etc.)')
    parser.add_argument('--epochs', type=int, help='Epoch sayısı')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--fusion', type=str, choices=['early', 'late', 'attention'],
                       help='Fusion tipi')
    parser.add_argument('--approach', type=str, choices=['bilateral', 'multi_view'],
                       help='Yaklaşım')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau', 'onecycle'],
                       help='Scheduler tipi')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Pretrained weights kullanma')
    parser.add_argument('--no-oversampling', action='store_true',
                       help='Oversampling kullanma')
    
    # MLflow
    parser.add_argument('--experiment-name', type=str, help='MLflow experiment ismi')
    parser.add_argument('--run-name', type=str, help='MLflow run ismi')
    
    args = parser.parse_args()
    
    # Config seç
    if args.config == 'fast_test':
        config = FastTestConfig
    elif args.config == 'small_data':
        config = SmallDataConfig
    elif args.config == 'large_data':
        config = LargeDataConfig
    elif args.config == 'swin':
        config = SwinConfig
    else:
        config = Config
    
    # Override'lar uygula
    if args.model:
        config.MODEL_NAME = args.model
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        # Manuel LR override
        config.LEARNING_RATE = args.lr
    if args.fusion:
        config.FUSION_TYPE = args.fusion
    if args.approach:
        config.APPROACH = args.approach
    if args.scheduler:
        config.SCHEDULER_TYPE = args.scheduler
    if args.no_pretrained:
        config.PRETRAINED = False
    if args.no_oversampling:
        config.USE_OVERSAMPLING = False
    if args.experiment_name:
        config.MLFLOW_EXPERIMENT_NAME = args.experiment_name
    
    return config, args

def get_dataloaders(config):
    """
    DataLoader'ları oluşturur
    Output olarak train_loader ve val_loader döner
    """
    train_transform = get_transforms(config, is_train = True, aggressive = False)
    val_transform = get_transforms(config, is_train = False)
    
    # Train dataseti
    train_dataset = BilateralMammogramDataset(
        config = config,
        class_folders = config.TRAIN_CLASSES,
        transform = train_transform,
        is_train = True
    )
    
    # Test dataset (validation)
    val_dataset = BilateralMammogramDataset(
        config = config,
        class_folders = config.TEST_CLASSES,
        transform = val_transform,
        is_train = False
    )

    # Boş dataset kontrolü
    if len(train_dataset) == 0:
        raise ValueError(
            f"Train dataset boş! Data klasörlerini kontrol edin.\n"
            f"Beklenen path: {config.DATA_ROOT}\n"
            f"Beklenen klasörler: {list(config.TRAIN_CLASSES.values())}"
        )

    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset boş! Data klasörlerini kontrol edin.\n"
            f"Beklenen path: {config.DATA_ROOT}\n"
            f"Beklenen klasörler: {list(config.TEST_CLASSES.values())}"
        )

    # Train dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        pin_memory = True if torch.cuda.is_available() else False
    )
    
    # Test dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = True if torch.cuda.is_available() else False
    )
    
    print(f"Train: {len(train_dataset)} örnek, {len(train_loader)} batch")
    print(f"Val: {len(val_dataset)} örnek, {len(val_loader)} batch")
    
    return train_loader, val_loader, train_dataset

def get_class_weights(train_dataset, config):
    """
    Class weightleri hesaplar
    """
    
    if not config.USE_CLASS_WEIGHTS:
        return None
    
    class_dist = train_dataset.get_class_distribution()
    
    if not class_dist:
        return None
    
    max_count = max(class_dist.values())
    weights = [max_count / class_dist.get(i ,max_count) for i in range(config.NUM_CLASSES)]
    
    weights_tensor = torch.FloatTensor(weights).to(config.DEVICE)
    
    print(f"Class Weights:")
    birads_mapping = {0: 1, 1: 2, 2: 4, 3: 5}
    
    for i, w in enumerate(weights):
        birads = birads_mapping[i]
        print(f"BI-RADS {birads}: {w:.3f}")
    return weights_tensor

def get_model_and_optimizer(config):
    """
    Modeli ve optimizeri oluşturur
    
    Model - optimizer - scheduler return eder
    """
    print(f"Model oluşturuluyor: {config.FUSION_TYPE} fusion")
    
    # Model seçimi
    # Model seç
    if config.FUSION_TYPE == "early":
        model = get_model(config)
    elif config.FUSION_TYPE == "late":
        model = get_late_fusion_model(config)
    elif config.FUSION_TYPE == "attention":
        model = get_attention_fusion_model(config)
    else:
        raise ValueError(f"Bilinmeyen fusion type: {config.FUSION_TYPE}")
    
    model = model.to(config.DEVICE)
    
    # Learning rate
    if hasattr(config, 'LEARNING_RATE'):
        # Manuel override varsa
        lr = config.LEARNING_RATE
    else:
        # Model-specific LR
        lr = config.get_learning_rate(config.MODEL_NAME)
    
    print(f"   Learning Rate: {lr}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = None
    if config.SCHEDULER_TYPE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=lr * 0.01
        )
        print(f"   Scheduler: Cosine Annealing")
    
    elif config.SCHEDULER_TYPE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print(f"   Scheduler: ReduceLROnPlateau")
    
    elif config.SCHEDULER_TYPE == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=1,  # Epoch bazlı
            pct_start=0.3
        )
        print(f"   Scheduler: OneCycleLR")
    
    return model, optimizer, scheduler, lr

def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch):
    """
    Bir epoch eğitim
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    
    for batch_idx, batch_data in enumerate(pbar):
        
        # Batch parse et
        if config.FUSION_TYPE == "early":
            images, labels = batch_data
            images = images.to(config.DEVICE)
        else:  # late veya attention
            cc_img, mlo_img, labels = batch_data
            cc_img = cc_img.to(config.DEVICE)
            mlo_img = mlo_img.to(config.DEVICE)
        
        labels = labels.to(config.DEVICE)
        
        # Forward
        optimizer.zero_grad()
        
        if config.FUSION_TYPE == "early":
            outputs = model(images)
        else:
            outputs = model(cc_img, mlo_img)
        
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Epoch sonucu
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, config):
    """
    Validation işlevini halleder
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            
            # Batch parse
            if config.FUSION_TYPE == "early":
                images, labels = batch_data
                images = images.to(config.DEVICE)
            else:
                cc_img, mlo_img, labels = batch_data
                cc_img = cc_img.to(config.DEVICE)
                mlo_img = mlo_img.to(config.DEVICE)
            
            labels = labels.to(config.DEVICE)
            
            # Forward
            if config.FUSION_TYPE == "early":
                outputs = model(images)
            else:
                outputs = model(cc_img, mlo_img)
            
            loss = criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Metrikleri hesapla
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    target_names = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 4', 'BI-RADS 5']
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1, cm, report


def train_model(config, run_name=None):
    """
    Ana train fonksiyonu
    """
    # Config yazdır
    config.print_config()
    
    # MLflow setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # Run name
    if run_name is None:
        run_name = f"{config.MODEL_NAME}_{config.FUSION_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        
        # Parametreleri loglar
        mlflow.log_params({
            'model_name': config.MODEL_NAME,
            'fusion_type': config.FUSION_TYPE,
            'approach': config.APPROACH,
            'img_size': config.IMG_SIZE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.get_learning_rate(config.MODEL_NAME) if not hasattr(config, 'LEARNING_RATE') else config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'scheduler': config.SCHEDULER_TYPE,
            'pretrained': config.PRETRAINED,
            'use_oversampling': config.USE_OVERSAMPLING,
            'oversampling_strategy': config.OVERSAMPLING_STRATEGY if config.USE_OVERSAMPLING else None,
            'use_class_weights': config.USE_CLASS_WEIGHTS,
            'label_smoothing': config.USE_LABEL_SMOOTHING,
        })
        
        # DataLoaders
        train_loader, val_loader, train_dataset = get_dataloaders(config)
        
        # Class weights
        class_weights = get_class_weights(train_dataset, config)
        
        # Loss function
        if config.USE_LABEL_SMOOTHING:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.LABEL_SMOOTHING_FACTOR
            )
            print(f"\n Loss: CrossEntropyLoss + Label Smoothing ({config.LABEL_SMOOTHING_FACTOR})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"\n Loss: CrossEntropyLoss")
        
        # Model, optimizer, scheduler
        model, optimizer, scheduler, lr = get_model_and_optimizer(config)
        
        # Checkpoint klasörü
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_acc = 0.0
        best_report = None
        patience_counter = 0
        
        print(f"\n{'='*70}")
        print(" TRAINING")
        print('='*70)
        
        for epoch in range(config.NUM_EPOCHS):
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, config, epoch
            )
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, cm, report = validate(
                model, val_loader, criterion, config
            )
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"   Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
            
            # Scheduler step
            if scheduler:
                if config.SCHEDULER_TYPE == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Best model kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_report = report
                patience_counter = 0

                # Checkpoint kaydet
                checkpoint_path = config.CHECKPOINT_DIR / f"best_{run_name}.pth"
                # Config'den sadece serializable değerleri al
                config_dict = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(config).items()
                    if not k.startswith('_') and not callable(getattr(config, k))
                }
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'config': config_dict
                }, checkpoint_path)

                print(f"Best model kaydedildi: {val_acc:.4f}")

                # MLflow'a kaydet
                mlflow.pytorch.log_model(model, "best_model")
                mlflow.log_text(report, "classification_report.txt")
            
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{config.PATIENCE}")
            
            # Early stopping
            if patience_counter >= config.PATIENCE:
                print(f"\n Early stopping: {config.PATIENCE} epoch boyunca iyileşme yok.")
                break
        
        # Final results
        print(f"\n{'='*70}")
        print("Train tamamlandı")
        print('='*70)
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")

        # Classification report yazdır
        if best_report:
            print(f"\n Classification Report (Best Model):")
            print(best_report)

        # Confusion matrix logla
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        return best_val_acc


if __name__ == "__main__":
    config, args = get_config_from_args()
    best_acc = train_model(config, run_name=args.run_name)
    print(f"\n🎉 Best Accuracy: {best_acc:.4f}")