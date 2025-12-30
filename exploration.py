import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DIABETIC RETINOPATHY DATASET EXPLORATION")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND INSPECT TRAINING LABELS
# ============================================================================
print("\n" + "="*70)
print("STEP 1: BASIC DATASET INSPECTION")
print("="*70)

train_df = pd.read_csv('train.csv')

print(f"\nDataset Shape: {train_df.shape}")
print(f"Total Images: {len(train_df)}")
print("\nFirst 10 rows:")
print(train_df.head(10))

print("\nColumn Info:")
print(train_df.dtypes)

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nBasic Statistics:")
print(train_df.describe())

# ============================================================================
# STEP 2: CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: CLASS DISTRIBUTION ANALYSIS")
print("="*70)

class_counts = train_df['diagnosis'].value_counts().sort_index()
print("\nClass Distribution (Counts):")
print(class_counts)

class_percentages = (class_counts / len(train_df) * 100).round(2)
print("\nClass Distribution (%):")
for cls, pct in class_percentages.items():
    print(f"  Class {cls}: {class_counts[cls]:5d} images ({pct:5.2f}%)")

max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class
print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}x")
print(f"Most common class: {class_counts.idxmax()} ({class_counts.max()} images)")
print(f"Rarest class: {class_counts.idxmin()} ({class_counts.min()} images)")

# Visualization
fig = plt.figure(figsize=(16, 10))

# Bar plot
ax1 = plt.subplot(2, 2, 1)
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1, palette='viridis')
ax1.set_xlabel('DR Severity Level', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax1.set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
for i, v in enumerate(class_counts.values):
    ax1.text(i, v + max_class*0.02, str(v), ha='center', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')

# Pie chart
ax2 = plt.subplot(2, 2, 2)
colors = sns.color_palette('viridis', len(class_counts))
ax2.pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], 
        autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 11})
ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

# ============================================================================
# STEP 3: IMAGE VISUALIZATION BY CLASS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: VISUALIZING SAMPLE IMAGES BY CLASS")
print("="*70)

severity_labels = [
    "0: No DR",
    "1: Mild DR", 
    "2: Moderate DR",
    "3: Severe DR",
    "4: Proliferative DR"
]

num_samples = 3
axes_list = []
for severity in range(5):
    print(f"Loading Class {severity} samples...")
    severity_images = train_df[train_df['diagnosis'] == severity].sample(
        min(num_samples, len(train_df[train_df['diagnosis'] == severity])),
        random_state=42
    )
    
    for idx, (_, row) in enumerate(severity_images.iterrows()):
        if idx >= num_samples:
            break
            
        img_path = os.path.join('train_images', f"{row['id_code']}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = plt.subplot(5, num_samples, severity * num_samples + idx + 1)
            ax.imshow(img)
            ax.axis('off')
            if idx == 0:
                ax.set_title(f"{severity_labels[severity]}", 
                           fontsize=11, fontweight='bold', loc='left')
            axes_list.append(ax)

plt.suptitle('Sample Retinal Images by DR Severity Level', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('class_distribution_and_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: class_distribution_and_samples.png")

# ============================================================================
# STEP 4: IMAGE METADATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: IMAGE PROPERTIES ANALYSIS")
print("="*70)

sample_size = 500
sample_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)

properties = {
    'heights': [],
    'widths': [],
    'channels': [],
    'file_sizes_mb': [],
    'mean_brightness': [],
    'std_brightness': []
}

print(f"Analyzing {len(sample_df)} random images for properties...")

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing"):
    img_path = os.path.join('train_images', f"{row['id_code']}.png")
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        
        if img is not None:
            h, w, c = img.shape
            properties['heights'].append(h)
            properties['widths'].append(w)
            properties['channels'].append(c)
            
            file_size = os.path.getsize(img_path) / (1024 * 1024)
            properties['file_sizes_mb'].append(file_size)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            properties['mean_brightness'].append(gray.mean())
            properties['std_brightness'].append(gray.std())

print("\n--- IMAGE PROPERTIES STATISTICS ---")
for key, values in properties.items():
    if len(values) > 0:
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Std:  {np.std(values):.2f}")
        print(f"  Min:  {np.min(values):.2f}")
        print(f"  Max:  {np.max(values):.2f}")

# Visualization
fig2 = plt.figure(figsize=(16, 10))

plot_configs = [
    ('heights', 'Image Heights (pixels)', 'steelblue'),
    ('widths', 'Image Widths (pixels)', 'coral'),
    ('file_sizes_mb', 'File Sizes (MB)', 'green'),
    ('mean_brightness', 'Mean Brightness (0-255)', 'purple'),
    ('std_brightness', 'Brightness Std Dev', 'orange'),
]

for idx, (key, title, color) in enumerate(plot_configs):
    ax = plt.subplot(2, 3, idx + 1)
    ax.hist(properties[key], bins=30, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution: {title}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    mean_val = np.mean(properties[key])
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig('image_properties_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: image_properties_distribution.png")

# ============================================================================
# STEP 5: CLASS-WISE IMAGE QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: CLASS-WISE QUALITY ANALYSIS")
print("="*70)

sample_per_class = 100
quality_by_class = {i: {'brightness': [], 'contrast': []} for i in range(5)}

for severity in range(5):
    severity_df = train_df[train_df['diagnosis'] == severity].sample(
        min(sample_per_class, len(train_df[train_df['diagnosis'] == severity])),
        random_state=42
    )
    
    print(f"Analyzing Class {severity} quality ({len(severity_df)} images)...")
    
    for _, row in tqdm(severity_df.iterrows(), total=len(severity_df), 
                      desc=f"Class {severity}", leave=False):
        img_path = os.path.join('train_images', f"{row['id_code']}.png")
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                quality_by_class[severity]['brightness'].append(gray.mean())
                quality_by_class[severity]['contrast'].append(gray.std())

print("\n--- QUALITY METRICS BY CLASS ---")
for severity in range(5):
    if len(quality_by_class[severity]['brightness']) > 0:
        avg_brightness = np.mean(quality_by_class[severity]['brightness'])
        avg_contrast = np.mean(quality_by_class[severity]['contrast'])
        print(f"\nClass {severity} ({severity_labels[severity]}):")
        print(f"  Avg Brightness: {avg_brightness:.2f}")
        print(f"  Avg Contrast:   {avg_contrast:.2f}")

# Visualization
fig3 = plt.figure(figsize=(14, 6))

# Brightness by class
ax1 = plt.subplot(1, 2, 1)
brightness_means = [np.mean(quality_by_class[i]['brightness']) for i in range(5)]
brightness_stds = [np.std(quality_by_class[i]['brightness']) for i in range(5)]
ax1.bar(range(5), brightness_means, yerr=brightness_stds, color='coral', 
        edgecolor='black', alpha=0.8, capsize=5)
ax1.set_xlabel('DR Severity Level', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Brightness (0-255)', fontsize=12, fontweight='bold')
ax1.set_title('Average Image Brightness by Class', fontsize=14, fontweight='bold')
ax1.set_xticks(range(5))
ax1.set_xticklabels([f'Class {i}' for i in range(5)])
ax1.grid(alpha=0.3, axis='y')

# Contrast by class
ax2 = plt.subplot(1, 2, 2)
contrast_means = [np.mean(quality_by_class[i]['contrast']) for i in range(5)]
contrast_stds = [np.std(quality_by_class[i]['contrast']) for i in range(5)]
ax2.bar(range(5), contrast_means, yerr=contrast_stds, color='skyblue', 
        edgecolor='black', alpha=0.8, capsize=5)
ax2.set_xlabel('DR Severity Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Contrast (Std Dev)', fontsize=12, fontweight='bold')
ax2.set_title('Average Image Contrast by Class', fontsize=14, fontweight='bold')
ax2.set_xticks(range(5))
ax2.set_xticklabels([f'Class {i}' for i in range(5)])
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('quality_by_class.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: quality_by_class.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EXPLORATION SUMMARY")
print("="*70)

print(f"""
DATASET OVERVIEW:
- Total Images: {len(train_df)}
- Classes: 0 (No DR) to 4 (Proliferative DR)

CLASS DISTRIBUTION:
- Imbalance Ratio: {imbalance_ratio:.2f}x
- Most Common: Class {class_counts.idxmax()} ({class_counts.max()} images, {class_percentages[class_counts.idxmax()]:.1f}%)
- Rarest: Class {class_counts.idxmin()} ({class_counts.min()} images, {class_percentages[class_counts.idxmin()]:.1f}%)

IMAGE PROPERTIES:
- Dimensions: {np.mean(properties['heights']):.0f}×{np.mean(properties['widths']):.0f} (±{np.std(properties['heights']):.0f}×{np.std(properties['widths']):.0f})
- File Size: {np.mean(properties['file_sizes_mb']):.2f} MB (±{np.std(properties['file_sizes_mb']):.2f})
- Brightness: {np.mean(properties['mean_brightness']):.1f} (±{np.std(properties['mean_brightness']):.1f})
- Contrast: {np.mean(properties['std_brightness']):.1f} (±{np.std(properties['std_brightness']):.1f})

GENERATED FILES:
✓ class_distribution_and_samples.png
✓ image_properties_distribution.png  
✓ quality_by_class.png

KEY FINDINGS TO CONSIDER:
1. Class imbalance needs addressing (weighted loss, oversampling, focal loss)
2. Image dimensions {'consistent' if np.std(properties['heights']) < 50 else 'vary - need resizing'}
3. Brightness variation suggests need for normalization/augmentation
4. Quality differences across classes may indicate preprocessing requirements
""")

print("="*70)
print("EXPLORATION COMPLETE!")
print("="*70)

plt.show()