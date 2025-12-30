import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DIABETIC RETINOPATHY - PREPROCESSING PIPELINE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'IMAGE_SIZE': 512,
    'TRAIN_RATIO': 0.70,
    'VAL_RATIO': 0.15,
    'TEST_RATIO': 0.15,
    'CLAHE_CLIP_LIMIT': 2.0,
    'CLAHE_TILE_SIZE': 8,
    'RANDOM_STATE': 42,
    'INPUT_DIR': 'train_images',
    'OUTPUT_DIR': 'preprocessed_data',
    'CSV_FILE': 'train.csv'
}

print("\n📋 CONFIGURATION:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Create output directories
Path(CONFIG['OUTPUT_DIR']).mkdir(exist_ok=True)
Path(f"{CONFIG['OUTPUT_DIR']}/train").mkdir(exist_ok=True)
Path(f"{CONFIG['OUTPUT_DIR']}/val").mkdir(exist_ok=True)
Path(f"{CONFIG['OUTPUT_DIR']}/test").mkdir(exist_ok=True)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def crop_circle_and_resize(image, target_size=512):
    """
    Remove black borders (circular crop) and resize to target size.
    
    Why: Fundus images have circular field of view with black borders
    that waste computational resources and don't contain medical info.
    """
    # Convert to grayscale for mask detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create binary mask (anything above threshold is "image", below is "border")
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours to detect the circular region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the fundus circle)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop to bounding box
        cropped = image[y:y+h, x:x+w]
    else:
        # Fallback: use original image if no contour found
        cropped = image
    
    # Resize to target size using high-quality interpolation
    resized = cv2.resize(cropped, (target_size, target_size), 
                         interpolation=cv2.INTER_AREA)
    
    return resized


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Why: Retinal images often have uneven illumination. CLAHE enhances
    local contrast, making microaneurysms, hemorrhages, and exudates
    more visible to the CNN.
    
    Technical: Works on LAB color space to avoid color distortion.
    """
    # Convert BGR to LAB color space
    # L = Lightness, A = green-red, B = blue-yellow
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only (preserves color)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced


def normalize_image(image):
    """
    Normalize pixel values to [0, 1] range.
    
    Why: Neural networks train better with normalized inputs.
    Standard preprocessing for CNNs.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path, target_size=512, apply_clahe_flag=True):
    """
    Complete preprocessing pipeline for a single image.
    
    Steps:
    1. Load image
    2. Crop circular region and resize
    3. Apply CLAHE (if enabled)
    4. Normalize to [0, 1]
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Step 1: Crop and resize
    img = crop_circle_and_resize(img, target_size)
    
    # Step 2: Apply CLAHE
    if apply_clahe_flag:
        img = apply_clahe(img, 
                         clip_limit=CONFIG['CLAHE_CLIP_LIMIT'],
                         tile_size=CONFIG['CLAHE_TILE_SIZE'])
    
    # Step 3: Normalize
    img = normalize_image(img)
    
    return img


def visualize_preprocessing_effect(image_path, output_path='preprocessing_comparison.png'):
    """
    Visualize the effect of preprocessing steps.
    """
    # Load original
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Step by step
    cropped = crop_circle_and_resize(original, CONFIG['IMAGE_SIZE'])
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    clahe_applied = apply_clahe(cropped, CONFIG['CLAHE_CLIP_LIMIT'], 
                                CONFIG['CLAHE_TILE_SIZE'])
    clahe_rgb = cv2.cvtColor(clahe_applied, cv2.COLOR_BGR2RGB)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cropped_rgb)
    axes[1].set_title(f'Cropped & Resized ({CONFIG["IMAGE_SIZE"]}x{CONFIG["IMAGE_SIZE"]})', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(clahe_rgb)
    axes[2].set_title('CLAHE Applied', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved preprocessing visualization: {output_path}")
    plt.close()


# ============================================================================
# STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("STEP 1: STRATIFIED DATA SPLITTING")
print("="*70)

# Load dataset
df = pd.read_csv(CONFIG['CSV_FILE'])
print(f"\nTotal samples: {len(df)}")

# First split: Train vs (Val + Test)
train_df, temp_df = train_test_split(
    df,
    test_size=(CONFIG['VAL_RATIO'] + CONFIG['TEST_RATIO']),
    stratify=df['diagnosis'],
    random_state=CONFIG['RANDOM_STATE']
)

# Second split: Val vs Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=CONFIG['TEST_RATIO'] / (CONFIG['VAL_RATIO'] + CONFIG['TEST_RATIO']),
    stratify=temp_df['diagnosis'],
    random_state=CONFIG['RANDOM_STATE']
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Verify stratification
print("\n📊 Class distribution across splits:")
splits_data = []
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['diagnosis'].value_counts().sort_index()
    pct = (dist / len(split_df) * 100).round(1)
    print(f"\n{split_name}:")
    for cls in range(5):
        print(f"  Class {cls}: {dist.get(cls, 0):4d} ({pct.get(cls, 0):4.1f}%)")
    splits_data.append({'split': split_name, 'dist': dist, 'pct': pct})

# Visualize split distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, split_info in enumerate(splits_data):
    axes[idx].bar(range(5), [split_info['dist'].get(i, 0) for i in range(5)],
                  color='steelblue', edgecolor='black', alpha=0.8)
    axes[idx].set_xlabel('DR Severity', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[idx].set_title(f"{split_info['split']} Set Distribution", 
                       fontsize=14, fontweight='bold')
    axes[idx].set_xticks(range(5))
    axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('split_distributions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved split visualization: split_distributions.png")
plt.close()

# Save split indices
train_df.to_csv(f"{CONFIG['OUTPUT_DIR']}/train_split.csv", index=False)
val_df.to_csv(f"{CONFIG['OUTPUT_DIR']}/val_split.csv", index=False)
test_df.to_csv(f"{CONFIG['OUTPUT_DIR']}/test_split.csv", index=False)
print("\n✓ Saved split CSV files")

# ============================================================================
# VISUALIZE PREPROCESSING EFFECT
# ============================================================================
print("\n" + "="*70)
print("STEP 2: VISUALIZING PREPROCESSING EFFECT")
print("="*70)

# Get a sample image from each class
sample_images = []
for cls in range(5):
    cls_samples = train_df[train_df['diagnosis'] == cls]
    if len(cls_samples) > 0:
        sample = cls_samples.sample(1, random_state=42).iloc[0]
        sample_images.append(sample)

# Visualize preprocessing for samples
fig, axes = plt.subplots(5, 3, figsize=(12, 20))

for idx, sample in enumerate(sample_images):
    img_path = os.path.join(CONFIG['INPUT_DIR'], f"{sample['id_code']}.png")
    
    # Original
    original = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Cropped
    cropped = crop_circle_and_resize(original, CONFIG['IMAGE_SIZE'])
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # CLAHE
    clahe_applied = apply_clahe(cropped, CONFIG['CLAHE_CLIP_LIMIT'], 
                                CONFIG['CLAHE_TILE_SIZE'])
    clahe_rgb = cv2.cvtColor(clahe_applied, cv2.COLOR_BGR2RGB)
    
    axes[idx, 0].imshow(original_rgb)
    axes[idx, 0].set_title(f"Class {sample['diagnosis']}: Original", fontsize=11)
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(cropped_rgb)
    axes[idx, 1].set_title(f"Cropped & Resized", fontsize=11)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(clahe_rgb)
    axes[idx, 2].set_title(f"CLAHE Applied", fontsize=11)
    axes[idx, 2].axis('off')

plt.suptitle('Preprocessing Pipeline: Original → Cropped → CLAHE', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('preprocessing_steps_all_classes.png', dpi=150, bbox_inches='tight')
print("✓ Saved preprocessing comparison: preprocessing_steps_all_classes.png")
plt.close()

# ============================================================================
# PROCESS AND SAVE ALL IMAGES
# ============================================================================
print("\n" + "="*70)
print("STEP 3: PROCESSING AND SAVING ALL IMAGES")
print("="*70)

def process_and_save_split(split_df, split_name):
    """Process and save all images for a given split."""
    print(f"\n🔄 Processing {split_name} set ({len(split_df)} images)...")
    
    output_dir = f"{CONFIG['OUTPUT_DIR']}/{split_name}"
    failed = []
    
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
        try:
            img_path = os.path.join(CONFIG['INPUT_DIR'], f"{row['id_code']}.png")
            
            # Preprocess
            processed_img = preprocess_image(
                img_path, 
                target_size=CONFIG['IMAGE_SIZE'],
                apply_clahe_flag=True
            )
            
            # Save as numpy array (efficient for training)
            output_path = os.path.join(output_dir, f"{row['id_code']}.npy")
            np.save(output_path, processed_img)
            
        except Exception as e:
            failed.append((row['id_code'], str(e)))
    
    if failed:
        print(f"⚠️  Failed to process {len(failed)} images:")
        for img_id, error in failed[:5]:  # Show first 5
            print(f"    {img_id}: {error}")
    else:
        print(f"✓ Successfully processed all {len(split_df)} images")
    
    return failed

# Process all splits
train_failed = process_and_save_split(train_df, 'train')
val_failed = process_and_save_split(val_df, 'val')
test_failed = process_and_save_split(test_df, 'test')

# ============================================================================
# CALCULATE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 4: CALCULATING CLASS WEIGHTS")
print("="*70)

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights for training set
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis']),
    y=train_df['diagnosis']
)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("\n📊 Class weights for loss function:")
for cls, weight in class_weight_dict.items():
    print(f"  Class {cls}: {weight:.4f}")

# Visualize class weights
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(5), class_weights, color='coral', edgecolor='black', alpha=0.8)
ax.set_xlabel('DR Severity Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Class Weight', fontsize=12, fontweight='bold')
ax.set_title('Computed Class Weights for Imbalanced Data', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(5))
ax.grid(alpha=0.3, axis='y')

for i, (bar, weight) in enumerate(zip(bars, class_weights)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{weight:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('class_weights.png', dpi=150, bbox_inches='tight')
print("✓ Saved class weights visualization: class_weights.png")
plt.close()

# Save class weights
with open(f"{CONFIG['OUTPUT_DIR']}/class_weights.json", 'w') as f:
    json.dump(class_weight_dict, f, indent=2)
print("✓ Saved class weights to JSON")

# ============================================================================
# SAVE PREPROCESSING METADATA
# ============================================================================
metadata = {
    'config': CONFIG,
    'dataset_stats': {
        'total_images': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'num_classes': 5
    },
    'class_distribution': {
        'train': train_df['diagnosis'].value_counts().sort_index().to_dict(),
        'val': val_df['diagnosis'].value_counts().sort_index().to_dict(),
        'test': test_df['diagnosis'].value_counts().sort_index().to_dict()
    },
    'class_weights': class_weight_dict,
    'failed_images': {
        'train': len(train_failed),
        'val': len(val_failed),
        'test': len(test_failed)
    }
}

with open(f"{CONFIG['OUTPUT_DIR']}/preprocessing_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print("\n✓ Saved preprocessing metadata")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)

print(f"""
📁 OUTPUT STRUCTURE:
{CONFIG['OUTPUT_DIR']}/
├── train/              ({len(train_df)} images, .npy format)
├── val/                ({len(val_df)} images, .npy format)
├── test/               ({len(test_df)} images, .npy format)
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_weights.json
└── preprocessing_metadata.json

🎨 GENERATED VISUALIZATIONS:
✓ split_distributions.png
✓ preprocessing_steps_all_classes.png
✓ class_weights.png

⚙️ PREPROCESSING APPLIED:
✓ Circular crop (remove black borders)
✓ Resize to {CONFIG['IMAGE_SIZE']}×{CONFIG['IMAGE_SIZE']}
✓ CLAHE contrast enhancement
✓ Normalization to [0, 1]

📊 CLASS WEIGHTS COMPUTED:
  (Use these in your loss function to handle imbalance)
""")

for cls, weight in class_weight_dict.items():
    count = train_df[train_df['diagnosis'] == cls].shape[0]
    print(f"  Class {cls}: weight={weight:.4f} (n={count})")

print("\n" + "="*70)
print("READY FOR MODEL TRAINING!")
print("="*70)