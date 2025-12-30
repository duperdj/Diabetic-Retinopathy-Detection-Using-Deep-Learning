import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DIABETIC RETINOPATHY - TRANSFER LEARNING (ResNet50)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'DATA_DIR': 'preprocessed_data',
    'MODEL_SAVE_DIR': 'models',
    'RESULTS_DIR': 'results',
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 30,
    'LEARNING_RATE': 0.0001,  # Lower LR for transfer learning
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 10,
    'NUM_CLASSES': 5,
    'IMAGE_SIZE': 512,
    'NUM_WORKERS': 0,  # Changed from 4 to 0 to avoid Windows multiprocessing issues
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'FOCAL_LOSS_GAMMA': 2.0,  # Focal loss parameter
    'FOCAL_LOSS_ALPHA': None,  # Will use class weights
}

print("\n📋 CONFIGURATION:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

Path(CONFIG['MODEL_SAVE_DIR']).mkdir(exist_ok=True)
Path(CONFIG['RESULTS_DIR']).mkdir(exist_ok=True)

# ============================================================================
# FOCAL LOSS (Better for Imbalanced Data)
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Why Focal Loss?
    - Standard CrossEntropy treats all examples equally
    - Focal Loss focuses training on hard examples
    - Down-weights easy examples (well-classified)
    - Up-weights hard examples (misclassified)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - gamma: focusing parameter (higher = more focus on hard examples)
    - alpha: class weights
    
    For DR: This helps the model learn Classes 1, 3, 4 despite being rare.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# CUSTOM DATASET WITH AUGMENTATION
# ============================================================================
class DiabeticRetinopathyDataset(Dataset):
    """Enhanced dataset with stronger augmentation for transfer learning"""
    
    def __init__(self, csv_file, data_dir, augment=False):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']
        
        # Load preprocessed image
        img_path = Path(self.data_dir) / f"{img_id}.npy"
        image = np.load(img_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Apply augmentation
        if self.augment:
            image = self.augment_image(image)
        
        return image, label
    
    def augment_image(self, img):
        """
        Apply data augmentation:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (±15°)
        - Random brightness/contrast
        
        Why: Retinal images can be viewed from any angle, flipped, etc.
        This increases effective dataset size and reduces overfitting.
        """
        # Horizontal flip (50% chance)
        if torch.rand(1) > 0.5:
            img = torch.flip(img, [2])
        
        # Vertical flip (50% chance)
        if torch.rand(1) > 0.5:
            img = torch.flip(img, [1])
        
        # Random brightness adjustment (±10%)
        if torch.rand(1) > 0.5:
            brightness_factor = 0.9 + torch.rand(1) * 0.2
            img = img * brightness_factor
            img = torch.clamp(img, 0, 1)
        
        return img

# ============================================================================
# RESNET50 TRANSFER LEARNING MODEL
# ============================================================================
class ResNet50DR(nn.Module):
    """
    ResNet50 pretrained on ImageNet, adapted for DR classification.
    
    Architecture:
    - ResNet50 backbone (pretrained on ImageNet)
    - Replace final FC layer with custom classifier
    - Add dropout for regularization
    
    Transfer Learning Strategy:
    1. Load pretrained weights (learned from 1.2M ImageNet images)
    2. Freeze early layers (keep low-level features like edges, textures)
    3. Fine-tune later layers (adapt to retinal-specific patterns)
    4. Train new classifier head (5-class DR classification)
    
    Why ResNet50?
    - 50 layers deep (vs our 4-layer baseline)
    - Residual connections prevent vanishing gradients
    - Proven architecture for medical imaging
    - 25M parameters but most are pretrained
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50DR, self).__init__()
        
        # Load pretrained ResNet50
        try:
            self.resnet = models.resnet50(pretrained=pretrained)
            print("✓ Loaded pretrained weights from PyTorch servers")
        except Exception as e:
            print(f"⚠️  Could not download pretrained weights: {e}")
            print("⚠️  Training from scratch (will take longer and may be less accurate)")
            self.resnet = models.resnet50(pretrained=False)
        
        # Get number of features from last layer
        num_features = self.resnet.fc.in_features
        
        # Replace classifier with custom head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze early layers for initial training"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze only the classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    # Load class weights
    with open(f"{CONFIG['DATA_DIR']}/class_weights.json", 'r') as f:
        class_weights_dict = json.load(f)
        class_weights = torch.FloatTensor([class_weights_dict[str(i)] for i in range(5)])
        class_weights = class_weights.to(CONFIG['DEVICE'])
    
    print(f"\n✓ Loaded class weights: {class_weights.cpu().numpy()}")
    
    # Create datasets with augmentation
    train_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/train_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/train",
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/val_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/val",
        augment=False
    )
    
    test_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/test_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/test",
        augment=False
    )
    
    print(f"\n✓ Train set: {len(train_dataset)} images (with augmentation)")
    print(f"✓ Val set:   {len(val_dataset)} images")
    print(f"✓ Test set:  {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'],
                             shuffle=True, num_workers=CONFIG['NUM_WORKERS'],
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'],
                           shuffle=False, num_workers=CONFIG['NUM_WORKERS'],
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'],
                            shuffle=False, num_workers=CONFIG['NUM_WORKERS'],
                            pin_memory=True)
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("INITIALIZING RESNET50 MODEL")
    print("="*70)
    
    model = ResNet50DR(num_classes=CONFIG['NUM_CLASSES'], pretrained=True)
    model = model.to(CONFIG['DEVICE'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model: ResNet50 (Transfer Learning)")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Pretrained: Yes (ImageNet)")
    print(f"✓ Device: {CONFIG['DEVICE']}")
    
    # Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights, gamma=CONFIG['FOCAL_LOSS_GAMMA'])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'],
                          weight_decay=CONFIG['WEIGHT_DECAY'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['NUM_EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, CONFIG['DEVICE'])
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion,
                                          CONFIG['DEVICE'])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f"{CONFIG['MODEL_SAVE_DIR']}/resnet50_best.pth")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['PATIENCE']:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # ========================================================================
    # PLOT TRAINING HISTORY
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('ResNet50 Training History - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('ResNet50 Training History - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/resnet50_training_history.png", dpi=150)
    print(f"\n✓ Saved training history plot")
    
    # ========================================================================
    # EVALUATE ON TEST SET
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    checkpoint = torch.load(f"{CONFIG['MODEL_SAVE_DIR']}/resnet50_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded best model")
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader,
                                                            criterion, CONFIG['DEVICE'])
    
    print(f"\n📊 TEST SET RESULTS:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(test_labels, test_preds,
                               target_names=[f'Class {i}' for i in range(5)],
                               digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f'Class {i}' for i in range(5)],
               yticklabels=[f'Class {i}' for i in range(5)])
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - ResNet50 Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/resnet50_confusion_matrix.png", dpi=150)
    print("\n✓ Saved confusion matrix")
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    for i in range(5):
        class_mask = np.array(test_labels) == i
        if class_mask.sum() > 0:
            class_preds = np.array(test_preds)[class_mask]
            class_acc = (class_preds == i).sum() / class_mask.sum() * 100
            print(f"Class {i}: {class_acc:.2f}% ({(class_preds == i).sum()}/{class_mask.sum()})")
    
    # Save results
    results = {
        'model': 'ResNet50',
        'config': CONFIG,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'history': history,
        'classification_report': classification_report(test_labels, test_preds,
                                                       target_names=[f'Class {i}' for i in range(5)],
                                                       output_dict=True)
    }
    
    with open(f"{CONFIG['RESULTS_DIR']}/resnet50_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # COMPARISON WITH BASELINE
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    try:
        with open(f"{CONFIG['RESULTS_DIR']}/training_results.json", 'r') as f:
            baseline_results = json.load(f)
        
        print(f"\nBaseline CNN:")
        print(f"  Test Accuracy: {baseline_results['test_acc']:.2f}%")
        print(f"\nResNet50 (Transfer Learning):")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"\n🚀 Improvement: {test_acc - baseline_results['test_acc']:.2f}%")
    except:
        print("Baseline results not found for comparison")
    
    print("\n" + "="*70)
    print("ALL DONE! Check 'results/' folder for outputs")
    print("="*70)   