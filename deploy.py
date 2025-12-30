import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import io

# ============================================
# MODEL ARCHITECTURE (ResNet Transfer Learning)
# ============================================
class ResNet50DR(nn.Module):
    """
    Exact architecture matching your training code:
    - ResNet50 backbone
    - 2-layer classifier: 2048 -> 512 -> 5
    - Dropout after each layer
    """
    def __init__(self, num_classes=5, pretrained=False):
        super(ResNet50DR, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features  # 2048
        
        # Replace classifier with your exact architecture
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# ============================================
# GRAD-CAM IMPLEMENTATION
# ============================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class):
        self.model.eval()
        
        # Ensure input is on the same device as model
        device = next(self.model.parameters()).device
        input_image = input_image.to(device)
        
        output = self.model(input_image)
        
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        gradients = self.gradients[0].cpu()
        activations = self.activations[0].cpu()
        
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.numpy()

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def preprocess_image(image):
    """Apply CLAHE and circular crop to retinal image"""
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Store original
    original = img_array.copy()
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Circular crop (find retina region)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        cropped = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    else:
        cropped = enhanced
    
    # Resize to 512x512
    resized = cv2.resize(cropped, (512, 512))
    
    return original, enhanced, resized

def create_overlay(original_img, heatmap, alpha=0.4):
    """Create beautiful heatmap overlay on original image"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Apply colormap (jet is medical standard)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

# ============================================
# CLINICAL INTERPRETATION
# ============================================
def get_clinical_interpretation(pred_class, confidence):
    """Provide medical interpretation and recommendations"""
    
    interpretations = {
        0: {
            "name": "No Diabetic Retinopathy",
            "description": "No signs of diabetic retinopathy detected. The retina appears healthy with no visible microaneurysms, hemorrhages, or other DR-related abnormalities.",
            "severity": "✅ Normal",
            "recommendation": "Continue regular annual eye examinations. Maintain good blood sugar control and follow your diabetes management plan.",
            "color": "#10b981"  # Green
        },
        1: {
            "name": "Mild Diabetic Retinopathy",
            "description": "Early stage DR with microaneurysms present. Small areas of balloon-like swelling in the retina's tiny blood vessels are visible.",
            "severity": "⚠️ Mild",
            "recommendation": "Schedule follow-up examination in 6-12 months. Improve blood sugar control. No immediate treatment required, but monitoring is essential.",
            "color": "#fbbf24"  # Yellow
        },
        2: {
            "name": "Moderate Diabetic Retinopathy",
            "description": "Progression of DR with blocked blood vessels. More extensive microaneurysms, hemorrhages, and possibly cotton wool spots are present.",
            "severity": "⚠️ Moderate",
            "recommendation": "Ophthalmologist consultation within 3-6 months. Strict blood sugar and blood pressure control required. May need more frequent monitoring.",
            "color": "#f59e0b"  # Orange
        },
        3: {
            "name": "Severe Diabetic Retinopathy",
            "description": "Advanced DR with significant blood vessel blockage. Multiple hemorrhages, microaneurysms across retinal quadrants. High risk of progression.",
            "severity": "🚨 Severe",
            "recommendation": "URGENT: Ophthalmologist consultation within 1-2 weeks. High risk of vision loss. Laser treatment or injections may be needed. Immediate blood sugar management critical.",
            "color": "#ef4444"  # Red
        },
        4: {
            "name": "Proliferative Diabetic Retinopathy",
            "description": "Most advanced stage. New abnormal blood vessels growing on retina/optic nerve. Risk of vitreous hemorrhage, retinal detachment, and severe vision loss.",
            "severity": "🚨 CRITICAL",
            "recommendation": "IMMEDIATE ophthalmologist referral (within days). Requires prompt treatment with laser surgery, injections, or vitrectomy. Vision-threatening emergency. Do not delay.",
            "color": "#dc2626"  # Dark Red
        }
    }
    
    info = interpretations[pred_class]
    
    # Create formatted output
    output = f"""
    <div style='background: linear-gradient(135deg, {info['color']}15, {info['color']}05); 
                padding: 25px; border-radius: 15px; border-left: 5px solid {info['color']};'>
        <h2 style='color: {info['color']}; margin-top: 0;'>{info['severity']} - {info['name']}</h2>
        <p style='font-size: 16px; line-height: 1.6; color: #374151;'><strong>Clinical Findings:</strong> {info['description']}</p>
        <p style='font-size: 16px; line-height: 1.6; color: #374151;'><strong>Model Confidence:</strong> {confidence:.1f}%</p>
        <div style='background: white; padding: 15px; border-radius: 10px; margin-top: 15px;'>
            <p style='font-size: 16px; margin: 0; color: #1f2937;'><strong>📋 Recommendation:</strong> {info['recommendation']}</p>
        </div>
    </div>
    """
    
    return output

# ============================================
# MAIN PREDICTION FUNCTION
# ============================================
def predict_diabetic_retinopathy(image):
    """Main prediction pipeline"""
    
    try:
        # Load model - matching your exact training code
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet50DR(num_classes=5, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(
            r'C:\Projects\Diabetic Retinopathy Detection System\aptos2019-blindness-detection\models\resnet50_best.pth',
            map_location=device
        )
        
        # Load the saved state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Preprocess image
        original, enhanced, preprocessed = preprocess_image(image)
        
        # Prepare for model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(preprocessed).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item() * 100
        
        # Generate Grad-CAM
        target_layer = model.resnet.layer4[-1]
        grad_cam = GradCAM(model, target_layer)
        
        # Create a new tensor for Grad-CAM (requires gradients)
        cam_input = input_tensor.clone().requires_grad_(True)
        cam = grad_cam.generate_cam(cam_input, pred_class)
        
        # Create overlay
        overlay = create_overlay(preprocessed, cam, alpha=0.5)
        
        # Get clinical interpretation
        interpretation = get_clinical_interpretation(pred_class, confidence)
        
        # Create probability distribution
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        probs = probabilities[0].cpu().numpy()
        
        prob_dict = {class_names[i]: float(probs[i]) for i in range(5)}
        
        return original, enhanced, preprocessed, overlay, interpretation, prob_dict
    
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}", {}

# ============================================
# GRADIO INTERFACE
# ============================================

# Custom CSS for beautiful medical interface
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.medical-header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 30px;
}
.step-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
        <div class='medical-header'>
            <h1 style='margin: 0; font-size: 2.5em; font-weight: bold;'>🔬 Diabetic Retinopathy Detection System</h1>
            <p style='margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;'>AI-Powered Retinal Analysis with Explainable Deep Learning</p>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;'>ResNet-50 Architecture | Grad-CAM Visualization | Clinical-Grade Interpretation</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Retinal Image")
            input_image = gr.Image(type="pil", label="Retinal Fundus Image", height=400)
            predict_btn = gr.Button("🔍 Analyze Retina", variant="primary", size="lg")
            
            gr.Markdown("""
                ---
                **ℹ️ About This System:**
                - **Model**: ResNet-50 Transfer Learning
                - **Classes**: 5-level DR severity (0-4)
                - **XAI**: Grad-CAM attention visualization
                - **Dataset**: APTOS 2019 Blindness Detection
            """)
    
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            
            with gr.Tabs():
                with gr.Tab("🎯 Diagnosis"):
                    clinical_output = gr.HTML(label="Clinical Interpretation")
                    prob_plot = gr.Label(label="Confidence Distribution", num_top_classes=5)
                
                with gr.Tab("🔥 AI Attention Map"):
                    gr.Markdown("**Grad-CAM Heatmap:** Red regions indicate where the AI focused to make its decision")
                    heatmap_output = gr.Image(label="Model Attention Overlay", height=500)
                
                with gr.Tab("⚙️ Preprocessing Pipeline"):
                    with gr.Row():
                        original_output = gr.Image(label="1️⃣ Original Image")
                        enhanced_output = gr.Image(label="2️⃣ CLAHE Enhanced")
                        preprocessed_output = gr.Image(label="3️⃣ Cropped & Resized")
    
    # Connect button to function
    predict_btn.click(
        fn=predict_diabetic_retinopathy,
        inputs=[input_image],
        outputs=[original_output, enhanced_output, preprocessed_output, 
                heatmap_output, clinical_output, prob_plot]
    )
    
    gr.Markdown("""
        ---
        ### 🔬 Research Methodology
        
        **Model Architecture:**
        - Base: ResNet-50 pretrained on ImageNet
        - Custom classifier head with dropout regularization
        - Class-weighted loss function for imbalanced data
        
        **Explainable AI:**
        - Grad-CAM (Gradient-weighted Class Activation Mapping)
        - Visualizes spatial attention of convolutional layers
        - Helps clinicians understand AI decision-making
        
        **Clinical Validation:**
        - 5-class diabetic retinopathy severity grading
        - Follows international DR classification standards
        - Recommendations based on ophthalmology guidelines
        
        ---
        
        ⚠️ **Medical Disclaimer:** This is a research demonstration system. Always consult qualified ophthalmologists for actual medical diagnosis and treatment decisions.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)