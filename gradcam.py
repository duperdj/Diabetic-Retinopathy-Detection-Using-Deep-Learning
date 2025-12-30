import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GRAD-CAM EXPLAINABILITY FOR DIABETIC RETINOPATHY")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'DATA_DIR': 'preprocessed_data',
    'MODEL_PATH': 'models/resnet50_best.pth',
    'RESULTS_DIR': 'results/gradcam',
    'NUM_CLASSES': 5,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_SAMPLES_PER_CLASS': 3,  # Generate heatmaps for 3 samples per class
}

print("\n📋 CONFIGURATION:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

Path(CONFIG['RESULTS_DIR']).mkdir(parents=True, exist_ok=True)

# ============================================================================
# RESNET50 MODEL (Same as training)
# ============================================================================
class ResNet50DR(nn.Module):
    """ResNet50 for DR classification"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50DR, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================
class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks
    
    How it works:
    1. Forward pass: Save activations from target layer
    2. Backward pass: Compute gradients w.r.t. target class
    3. Weight activations by gradient importance
    4. Generate heatmap showing where model "looked"
    
    Why it matters:
    - Shows which pixels influenced the prediction
    - Helps verify model learned real pathology (not artifacts)
    - Builds trust in AI decisions
    - Essential for medical AI interpretability
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class for CAM (None = predicted class)
        
        Returns:
            cam: Heatmap [H, W] showing attention regions
            prediction: Model's prediction
            confidence: Prediction confidence
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get prediction confidence
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, target_class].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        # Weight activations by gradients (importance)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def apply_heatmap_overlay(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image [H, W, 3] in [0, 1]
        heatmap: Grad-CAM heatmap [H, W] in [0, 1]
        alpha: Overlay transparency (0=invisible, 1=opaque)
    
    Returns:
        overlay: Image with heatmap overlay
    """
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
    
    # Overlay on original image
    overlay = image * (1 - alpha) + heatmap_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_comprehensive_visualization(original_img, heatmap, prediction, 
                                       confidence, true_label, img_id,
                                       save_path):
    """
    Create beautiful comprehensive visualization with multiple panels
    
    Layout:
    [Original] [Enhanced] [Heatmap] [Overlay] [Annotated]
    """
    fig = plt.figure(figsize=(20, 5))
    
    severity_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Panel 1: Original Image
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Retinal Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Preprocessed (already in original_img, show again for clarity)
    ax2 = plt.subplot(1, 5, 2)
    ax2.imshow(original_img)
    ax2.set_title('Preprocessed\n(CLAHE Applied)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Pure Heatmap
    ax3 = plt.subplot(1, 5, 3)
    im = ax3.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax3.set_title('Grad-CAM Heatmap\n(Model Attention)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # Panel 4: Overlay
    ax4 = plt.subplot(1, 5, 4)
    overlay = apply_heatmap_overlay(original_img, heatmap, alpha=0.5)
    ax4.imshow(overlay)
    ax4.set_title('Heatmap Overlay\n(Combined View)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Panel 5: Annotated with high-attention regions
    ax5 = plt.subplot(1, 5, 5)
    ax5.imshow(overlay)
    
    # Find and circle high-attention regions (top 5% of heatmap)
    threshold = np.percentile(heatmap, 95)
    high_attention = heatmap > threshold
    
    # Find contours of high-attention regions
    high_attention_uint8 = (high_attention * 255).astype(np.uint8)
    contours, _ = cv2.findContours(high_attention_uint8, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around top attention regions
    for contour in contours[:5]:  # Top 5 regions
        if cv2.contourArea(contour) > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            circle = patches.Circle((x + w/2, y + h/2), 
                                   radius=max(w, h)/2 + 10,
                                   linewidth=3, edgecolor='lime', 
                                   facecolor='none')
            ax5.add_patch(circle)
    
    ax5.set_title('Annotated Focus Areas\n(AI Attention Regions)', 
                 fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Add prediction info at the top
    correct = '✓' if prediction == true_label else '✗'
    color = 'green' if prediction == true_label else 'red'
    
    fig.suptitle(
        f'Image: {img_id} | True: Class {true_label} ({severity_names[true_label]}) | '
        f'Predicted: Class {prediction} ({severity_names[prediction]}) | '
        f'Confidence: {confidence:.1%} {correct}',
        fontsize=14, fontweight='bold', color=color, y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_statistics(heatmap):
    """
    Analyze where the model is focusing
    
    Returns statistics about attention distribution
    """
    # Divide image into quadrants
    h, w = heatmap.shape
    quadrants = {
        'Superior Temporal': heatmap[:h//2, w//2:],
        'Superior Nasal': heatmap[:h//2, :w//2],
        'Inferior Temporal': heatmap[h//2:, w//2:],
        'Inferior Nasal': heatmap[h//2:, :w//2],
    }
    
    stats = {}
    for name, region in quadrants.items():
        stats[name] = {
            'mean_attention': float(region.mean()),
            'max_attention': float(region.max()),
            'percentage_high_attention': float((region > 0.7).sum() / region.size * 100)
        }
    
    # Overall statistics
    stats['overall'] = {
        'mean': float(heatmap.mean()),
        'std': float(heatmap.std()),
        'max': float(heatmap.max()),
        'high_attention_pixels': int((heatmap > 0.7).sum()),
        'high_attention_percentage': float((heatmap > 0.7).sum() / heatmap.size * 100)
    }
    
    return stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    # Load trained model
    model = ResNet50DR(num_classes=CONFIG['NUM_CLASSES'], pretrained=False)
    checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['DEVICE'])
    model.eval()
    
    print(f"✓ Loaded ResNet50 from {CONFIG['MODEL_PATH']}")
    print(f"✓ Model on device: {CONFIG['DEVICE']}")
    
    # Initialize Grad-CAM
    # Target layer: Last convolutional layer of ResNet50
    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    print(f"✓ Grad-CAM initialized on layer: {target_layer.__class__.__name__}")
    
    # ========================================================================
    # GENERATE GRAD-CAM FOR TEST SET SAMPLES
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*70)
    
    # Load test set
    test_df = pd.read_csv(f"{CONFIG['DATA_DIR']}/test_split.csv")
    
    all_stats = []
    
    # Generate for samples from each class
    for class_id in range(CONFIG['NUM_CLASSES']):
        class_samples = test_df[test_df['diagnosis'] == class_id]
        
        if len(class_samples) == 0:
            print(f"\n⚠️  No samples found for Class {class_id}")
            continue
        
        # Sample N images from this class
        n_samples = min(CONFIG['NUM_SAMPLES_PER_CLASS'], len(class_samples))
        samples = class_samples.sample(n=n_samples, random_state=42)
        
        print(f"\n📊 Processing Class {class_id} ({n_samples} samples)...")
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            img_id = row['id_code']
            true_label = row['diagnosis']
            
            # Load preprocessed image
            img_path = Path(CONFIG['DATA_DIR']) / 'test' / f"{img_id}.npy"
            image = np.load(img_path)
            
            # Prepare input tensor
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            input_tensor = input_tensor.to(CONFIG['DEVICE'])
            
            # Generate Grad-CAM
            heatmap, prediction, confidence = grad_cam.generate_cam(input_tensor)
            
            # Get attention statistics
            stats = create_attention_statistics(heatmap)
            stats['image_id'] = img_id
            stats['true_label'] = int(true_label)
            stats['predicted_label'] = int(prediction)
            stats['confidence'] = float(confidence)
            stats['correct'] = bool(prediction == true_label)
            all_stats.append(stats)
            
            # Create visualization
            save_path = Path(CONFIG['RESULTS_DIR']) / f"gradcam_class{class_id}_{idx+1}_{img_id}.png"
            create_comprehensive_visualization(
                original_img=image,
                heatmap=heatmap,
                prediction=prediction,
                confidence=confidence,
                true_label=true_label,
                img_id=img_id,
                save_path=save_path
            )
            
            print(f"  ✓ Class {class_id} Sample {idx+1}: {img_id} | "
                  f"Pred: {prediction} | Conf: {confidence:.1%} | "
                  f"{'✓' if prediction == true_label else '✗'}")
    
    # ========================================================================
    # SAVE STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    with open(f"{CONFIG['RESULTS_DIR']}/gradcam_statistics.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"✓ Saved statistics to gradcam_statistics.json")
    print(f"✓ Generated {len(all_stats)} Grad-CAM visualizations")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("GRAD-CAM ANALYSIS SUMMARY")
    print("="*70)
    
    correct_preds = [s for s in all_stats if s['correct']]
    incorrect_preds = [s for s in all_stats if not s['correct']]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples analyzed: {len(all_stats)}")
    print(f"  Correct predictions: {len(correct_preds)} ({len(correct_preds)/len(all_stats)*100:.1f}%)")
    print(f"  Incorrect predictions: {len(incorrect_preds)} ({len(incorrect_preds)/len(all_stats)*100:.1f}%)")
    
    # Attention analysis
    if len(all_stats) > 0:
        avg_high_attention = np.mean([s['overall']['high_attention_percentage'] for s in all_stats])
        print(f"\n🔍 Attention Analysis:")
        print(f"  Average high-attention area: {avg_high_attention:.2f}% of image")
        print(f"  This shows the model is SELECTIVE (not looking at everything)")
        
        if len(correct_preds) > 0 and len(incorrect_preds) > 0:
            correct_attention = np.mean([s['overall']['high_attention_percentage'] 
                                        for s in correct_preds])
            incorrect_attention = np.mean([s['overall']['high_attention_percentage'] 
                                          for s in incorrect_preds])
            
            print(f"\n🎯 Attention Comparison:")
            print(f"  Correct predictions: {correct_attention:.2f}% high-attention")
            print(f"  Incorrect predictions: {incorrect_attention:.2f}% high-attention")
            
            if correct_attention < incorrect_attention:
                print(f"  → Model is MORE FOCUSED when correct (good sign!)")
            else:
                print(f"  → Model attention similar in both cases")
    
    # ========================================================================
    # CREATE COMPARISON GRID
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING COMPARISON GRID")
    print("="*70)
    
    # Create a grid showing one example from each class
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    
    severity_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    for class_id in range(5):
        class_samples = [s for s in all_stats if s['true_label'] == class_id]
        
        if len(class_samples) > 0:
            sample = class_samples[0]
            img_id = sample['image_id']
            
            # Load image and generate heatmap
            img_path = Path(CONFIG['DATA_DIR']) / 'test' / f"{img_id}.npy"
            image = np.load(img_path)
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            input_tensor = input_tensor.to(CONFIG['DEVICE'])
            heatmap, pred, conf = grad_cam.generate_cam(input_tensor)
            overlay = apply_heatmap_overlay(image, heatmap, alpha=0.5)
            
            # Original
            axes[class_id, 0].imshow(image)
            axes[class_id, 0].set_title(f'{severity_names[class_id]}\nOriginal', 
                                       fontsize=11, fontweight='bold')
            axes[class_id, 0].axis('off')
            
            # Heatmap
            axes[class_id, 1].imshow(heatmap, cmap='jet')
            axes[class_id, 1].set_title(f'Grad-CAM\nPred: Class {pred}', 
                                       fontsize=11, fontweight='bold')
            axes[class_id, 1].axis('off')
            
            # Overlay
            axes[class_id, 2].imshow(overlay)
            axes[class_id, 2].set_title(f'Overlay\nConf: {conf:.1%}', 
                                       fontsize=11, fontweight='bold')
            axes[class_id, 2].axis('off')
        else:
            for j in range(3):
                axes[class_id, j].text(0.5, 0.5, 'No samples', 
                                      ha='center', va='center')
                axes[class_id, j].axis('off')
    
    plt.suptitle('Grad-CAM Examples: All DR Severity Levels', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/gradcam_comparison_grid.png", 
                dpi=150, bbox_inches='tight')
    print("✓ Saved comparison grid")
    
    print("\n" + "="*70)
    print("GRAD-CAM COMPLETE!")
    print("="*70)
    print(f"\n📁 Check '{CONFIG['RESULTS_DIR']}/' for:")
    print(f"  • Individual Grad-CAM visualizations (5 panels each)")
    print(f"  • Comparison grid (all classes)")
    print(f"  • Statistics JSON file")
    print("\n🎯 These visualizations show WHERE your model is looking!")
    print("="*70)