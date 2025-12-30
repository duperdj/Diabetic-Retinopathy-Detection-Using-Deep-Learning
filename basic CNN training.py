import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'DATA_DIR': 'preprocessed_data',
    'MODEL_SAVE_DIR': 'models',
    'RESULTS_DIR': 'results',
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 10,
    'NUM_CLASSES': 5,
    'IMAGE_SIZE': 512,
    'NUM_WORKERS': 4,  # Will be set to 0 on Windows if needed
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
class DiabeticRetinopathyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed retinal images.
    
    Why custom: Our images are saved as .npy files (preprocessed).
    This loader is optimized for our specific data format.
    """
    
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file: Path to CSV with image IDs and labels
            data_dir: Directory containing .npy files
            transform: Optional transforms (we'll use basic ones)
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image ID and label
        row = self.data.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']
        
        # Load preprocessed .npy file
        img_path = Path(self.data_dir) / f"{img_id}.npy"
        image = np.load(img_path)
        
        # Convert to PyTorch tensor and rearrange dimensions
        # NumPy: (H, W, C) -> PyTorch: (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# DATA AUGMENTATION (Training only)
# ============================================================================
class RandomRotation:
    """Apply random rotation (fundus images can be viewed from any angle)"""
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, img):
        # Random rotation between -degrees and +degrees
        if torch.rand(1) > 0.5:
            angle = (torch.rand(1) * 2 - 1) * self.degrees
            # Simple rotation using PyTorch's grid_sample
            # For production, use torchvision.transforms.functional.rotate
            return img
        return img

class RandomHorizontalFlip:
    """Flip image horizontally with 50% probability"""
    def __call__(self, img):
        if torch.rand(1) > 0.5:
            return torch.flip(img, [2])  # Flip along width dimension
        return img

# ============================================================================
# BASELINE CNN ARCHITECTURE
# ============================================================================
class BaselineCNN(nn.Module):
    """
    Custom CNN Architecture for Diabetic Retinopathy Classification
    
    Architecture Philosophy:
    - Start with small filters (3x3) to capture fine details (microaneurysms)
    - Progressive feature extraction with increasing depth
    - Batch normalization for stable training
    - Dropout for regularization
    - Global average pooling to reduce parameters
    
    Why this design:
    - Medical images need careful feature extraction
    - We have limited data (3662 images), so we avoid too many parameters
    - Each conv block doubles channels and halves spatial dimensions
    """
    
    def __init__(self, num_classes=5):
        super(BaselineCNN, self).__init__()
        
        # Conv Block 1: 512x512x3 -> 256x256x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        # Conv Block 2: 256x256x32 -> 128x128x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Conv Block 3: 128x128x64 -> 64x64x128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # Conv Block 4: 64x64x128 -> 32x32x256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )
        
        # Global Average Pooling: 32x32x256 -> 1x1x256
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        
        return x

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
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


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
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    """Main training function"""
    
    print("="*70)
    print("DIABETIC RETINOPATHY - BASELINE CNN TRAINING")
    print("="*70)
    
    print("\n📋 CONFIGURATION:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create directories
    Path(CONFIG['MODEL_SAVE_DIR']).mkdir(exist_ok=True)
    Path(CONFIG['RESULTS_DIR']).mkdir(exist_ok=True)
    
    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)

    # Load class weights
    with open(f"{CONFIG['DATA_DIR']}/class_weights.json", 'r') as f:
        class_weights_dict = json.load(f)
        class_weights = torch.FloatTensor([class_weights_dict[str(i)] for i in range(5)])
        class_weights = class_weights.to(CONFIG['DEVICE'])

    print(f"\n✓ Loaded class weights: {class_weights.cpu().numpy()}")

    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/train_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/train",
        transform=RandomHorizontalFlip()
    )

    val_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/val_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/val"
    )

    test_dataset = DiabeticRetinopathyDataset(
        csv_file=f"{CONFIG['DATA_DIR']}/test_split.csv",
        data_dir=f"{CONFIG['DATA_DIR']}/test"
    )

    print(f"\n✓ Train set: {len(train_dataset)} images")
    print(f"✓ Val set:   {len(val_dataset)} images")
    print(f"✓ Test set:  {len(test_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )

    # ============================================================================
    # INITIALIZE MODEL
    # ============================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = BaselineCNN(num_classes=CONFIG['NUM_CLASSES'])
    model = model.to(CONFIG['DEVICE'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model: BaselineCNN")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Device: {CONFIG['DEVICE']}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )

    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(CONFIG['NUM_EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['DEVICE']
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, CONFIG['DEVICE']
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f"{CONFIG['MODEL_SAVE_DIR']}/best_model.pth")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['PATIENCE']:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # ============================================================================
    # PLOT TRAINING HISTORY
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/training_history.png", dpi=150)
    print(f"\n✓ Saved training history plot")

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    # Load best model
    checkpoint = torch.load(f"{CONFIG['MODEL_SAVE_DIR']}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded best model")

    # Evaluate
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, CONFIG['DEVICE']
    )

    print(f"\n📊 TEST SET RESULTS:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        test_labels,
        test_preds,
        target_names=[f'Class {i}' for i in range(5)],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(5)],
                yticklabels=[f'Class {i}' for i in range(5)])
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/confusion_matrix.png", dpi=150)
    print("\n✓ Saved confusion matrix")

    # Save results
    results = {
        'config': CONFIG,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'history': history
    }

    with open(f"{CONFIG['RESULTS_DIR']}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("ALL DONE! Check 'results/' folder for outputs")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    main()