# Diabetic-Retinopathy-Detection-Using-Deep-Learning

**Overview**

Diabetic Retinopathy (DR) is a diabetes-related eye disease and one of the leading causes of preventable blindness worldwide. Early detection is critical, yet large-scale screening is limited by the availability of trained ophthalmologists.
This project presents a deep learning–based medical imaging system that automatically detects and classifies diabetic retinopathy severity from retinal fundus images, while also providing explainable AI (Grad-CAM) visualizations to make the model’s decisions interpretable and trustworthy.

The system is designed as a research-grade, end-to-end pipeline — from data preprocessing and model training to explainability and web-based deployment.

**Objectives**

- Automatically classify diabetic retinopathy into 5 severity stage
- Build a custom CNN tailored for medical imaging
- Handle severe class imbalance using weighted loss
- Provide Explainable AI (Grad-CAM) to visualize model attention
- Deploy the system with a modern, professional web UI
- Emphasize interpretability, robustness, and ethical AI

**DR Severity Classes**

Label	Stage
0	No Diabetic Retinopathy
1	Mild
2	Moderate
3	Severe
4	Proliferative

**📊 Dataset**

APTOS 2019 Blindness Detection Dataset
Retinal fundus images
Publicly available Kaggle dataset
Significant class imbalance (handled explicitly)
Preprocessing performed:
Image resizing to 512 × 512
Normalization
Conversion to .npy format for faster I/O
Train / validation / test splits
Computation of class weights

🏗️ System Architecture
Backend (Deep Learning)

Framework: PyTorch
Model: Custom CNN (from scratch)
Batch Normalization & Dropout
Global Average Pooling
Class-weighted CrossEntropy loss
Adam optimizer + LR scheduling
Early stopping & checkpointing

🔹 Explainability

Grad-CAM
Heatmap visualization over retinal images
Highlights regions influencing predictions
Improves trust and clinical interpretability

🔹 Frontend (Deployment)

Next.js (App Router)
Tailwind CSS
Generated using V0 by Vercel
Modern medical-grade dashboard UI
Image upload, prediction display, Grad-CAM visualization

🧪 Model Training Details

Input resolution: 512 × 512
Batch size: 16
Epochs: up to 50 (early stopping enabled)
Optimizer: Adam
Scheduler: ReduceLROnPlateau
Best model automatically saved based on validation accuracy
Model checkpoints are stored at:
models/best_model.pth

📈 Evaluation Metrics

Accuracy
Precision, Recall, F1-Score
Confusion Matrix
Per-class performance analysis
Medical emphasis is placed on:
Early-stage detection
False-negative reduction
Generalization reliability

🔍 Explainable AI (Grad-CAM)

Instead of treating the model as a black box, this project integrates Grad-CAM to:
Visualize regions the CNN focuses on
Compare attention with known retinal lesions
Identify unreliable or overconfident predictions
Improve clinical trustworthiness
This makes the system suitable for assistive diagnosis, not blind automation.

⚖️ Ethical & Design Considerations

Predictions are based only on image data
Demographic data is not used as model input
Demographics are intended only for post-prediction analysis
The system is designed to assist clinicians, not replace them
Clear medical disclaimers included in UI

🖥️ Project Structure
├── backend/
│   ├── model/
│   ├── training/
│   ├── gradcam/
│   └── inference/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   └── styles/
│
├── data/
│   └── aptos2019/
│
├── models/
│   └── best_model.pth
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── metrics.json
│
└── README.md

🚀 Future Work

Transfer learning (ResNet / EfficientNet)
Cross-dataset generalization (IDRiD testing)
Early-stage DR sensitivity optimization
Human-AI disagreement analysis
Clinical risk stratification
Deployment in low-resource screening environments

⚠️ Disclaimer

This project is intended for research and educational purposes only.
It is not a medical device and should not be used for clinical diagnosis without professional supervision.

👤 Author

[Dhananjai Singh]
Computer Science & Engineering
Focus Areas:
  Deep Learning
  Medical Imaging
  Explainable AI
  Applied Machine Learning
