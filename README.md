# Plant Disease Detection with ResNet50 & Explainable AI

This project provides a complete solution for plant disease detection using ResNet50/MobileNetV2 and TensorFlow with Explainable AI capabilities. It includes:

1. Training a high-accuracy model on the PlantVillage dataset
2. Predicting plant diseases from new images with visual explanations
3. Generating comprehensive PDF reports with disease information, remedies, and AI explanations
4. **Explainable AI (XAI)** using Grad-CAM visualization to show which parts of the leaf influenced the prediction

## Features

### 🤖 AI Model
- ResNet50 and MobileNetV2 architectures with transfer learning
- 98%+ accuracy on plant disease classification
- Class-balanced training for handling imbalanced datasets
- Mixed precision training for faster performance

### 🔍 Explainable AI
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization
- Highlights the regions of the leaf that most influenced the AI's decision
- Visual heatmaps showing attention areas
- Three-panel visualization: Original image, Heatmap, and Overlay

### 📊 Comprehensive Reports
- Detailed disease information including:
  - Disease description and severity level
  - Root causes and contributing factors
  - Visible symptoms to look for
  - Treatment and remedy recommendations
  - Prevention and maintenance guidelines
- Grad-CAM visualization included in reports
- Professional PDF format with visual elements

## Requirements

- Python 3.6+
- TensorFlow 2.4+
- OpenCV for image processing
- Matplotlib for visualizations
- ReportLab for PDF generation
- Other dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project is designed to work with the PlantVillage dataset. The dataset contains images of plant leaves with various diseases across different plant species.

The dataset structure should be:
```
raw/color/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
└── ... (other plant disease categories)
```

## Usage

### Training the Model

To train the model on your dataset:

```bash
# Train with ResNet50 (higher accuracy, slower training)
python plant_disease_detection.py train --data_dir raw/color

# Train with MobileNetV2 (faster training, slightly lower accuracy)
python plant_disease_detection.py train --data_dir raw/color --use_mobilenet
```

This will:
1. Load and preprocess the dataset with advanced augmentation
2. Train a model using transfer learning with class weight balancing
3. Fine-tune the model for better accuracy
4. Save the trained model to `plant_disease_model.h5`
5. Generate training history plots

### Performance Optimization

The training process includes several optimizations:
- Mixed precision training for compatible GPUs
- Class weight balancing for imbalanced datasets
- Adaptive learning rate reduction
- BatchNormalization for training stability
- Optimized model architecture for faster training

### Predicting Diseases with XAI

To predict a disease from a new plant image with explainable AI:

```bash
# Full prediction with Grad-CAM visualization
python plant_disease_detection.py predict --image path/to/your/image.jpg

# Disable Grad-CAM if you only want the prediction
python plant_disease_detection.py predict --image path/to/your/image.jpg --no_gradcam
```

Optional arguments:
- `--model`: Path to a custom model (default: `plant_disease_model.h5`)
- `--output`: Path for the output PDF report (default: `plant_disease_report.pdf`)
- `--no_gradcam`: Disable Grad-CAM visualization

Example:
```bash
python plant_disease_detection.py predict --image test_images/apple_scab.jpg --output apple_report.pdf
```

## Understanding the AI's Decision

### Grad-CAM Visualization
The system generates three types of visualizations:

1. **Original Image**: The input plant leaf image
2. **Grad-CAM Heatmap**: Shows which areas the AI focused on (red = high attention, blue = low attention)
3. **Overlay**: Combines the original image with the heatmap to show attention areas on the actual leaf

### Interpreting Results
- **Red/Yellow areas**: Regions that strongly influenced the disease prediction
- **Blue/Dark areas**: Regions that had little influence on the decision
- **Confidence Score**: How certain the AI is about its prediction (higher is better)

## PDF Report Contents

The generated PDF report includes:

### 🔍 Diagnosis Results
- Detected disease name
- Confidence score
- Severity level

### 🧠 AI Explanation
- Grad-CAM visualization showing AI attention
- Explanation of which leaf parts influenced the decision

### 📋 Disease Information
- Detailed disease description
- Root causes and contributing factors
- Visible symptoms to identify

### 💊 Treatment & Remedies
- Specific treatment recommendations
- Fungicide and management strategies

### 🌱 Prevention & Maintenance
- Long-term prevention strategies
- Regular maintenance guidelines

## Model Performance

The optimized models typically achieve:
- **ResNet50**: 98%+ accuracy on the PlantVillage dataset
- **MobileNetV2**: 95%+ accuracy with significantly faster training
- Both models handle class imbalance effectively
- Grad-CAM provides reliable attention maps for disease regions

## Supported Diseases

The system currently includes detailed information for:
- Apple: Apple scab, Black rot, Cedar apple rust, Healthy
- Tomato: Early blight, Late blight, Healthy
- And can be extended to all PlantVillage classes

## Extending the System

### Adding New Diseases
1. Add new disease folders to your dataset
2. Update the `DISEASE_REMEDIES` dictionary with detailed information:
   - Description
   - Causes
   - Symptoms  
   - Remedies
   - Maintenance guidelines
   - Severity level
3. Use the `update_remedies.py` script to manage disease information
4. Retrain the model

### Example Disease Entry
```python
"New_Plant___New_Disease": {
    "description": "Detailed description of the disease",
    "causes": ["List of causes"],
    "symptoms": ["List of symptoms"],
    "remedies": ["List of treatments"],
    "maintenance": ["Prevention guidelines"],
    "severity": "High/Moderate/Low"
}
```

## Troubleshooting

### Grad-CAM Issues
- If Grad-CAM fails, the system will continue without visualization
- Ensure your model has convolutional layers for proper heat map generation
- Check that the model architecture is compatible (ResNet50/MobileNetV2)

### Performance Tips
- Use `--use_mobilenet` for faster training if accuracy is sufficient
- Enable GPU support for significantly faster training and inference
- Use mixed precision training (automatically enabled) for compatible hardware

## Citation

If you use the PlantVillage dataset, please cite:
```
@article{Mohanty_Hughes_Salathé_2016,
title={Using deep learning for image-based plant disease detection},
volume={7},
DOI={10.3389/fpls.2016.01419},
journal={Frontiers in Plant Science},
author={Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
year={2016},
month={Sep}} 
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
