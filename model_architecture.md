# Plant Disease Detection Model Architecture

This document explains the architecture of the plant disease detection model and the overall workflow of the system.

## Model Architecture

The model uses a ResNet50 architecture with transfer learning, which is a proven approach for image classification tasks. Here's a breakdown of the architecture:

### Base Model: ResNet50
- Pre-trained on ImageNet dataset (1.2 million images, 1000 classes)
- Deep residual learning framework with 50 layers
- Uses skip connections to address the vanishing gradient problem
- Excellent feature extraction capabilities for various image recognition tasks

### Custom Classification Layers
```
ResNet50 Base (Frozen during initial training)
↓
Global Average Pooling 2D
↓
Dense Layer (1024 units, ReLU activation)
↓
Dropout (0.2)
↓
Dense Layer (512 units, ReLU activation)
↓
Dropout (0.2)
↓
Output Layer (38 units, Softmax activation)
```

### Training Strategy
1. **Initial Training Phase**:
   - Freeze the base ResNet50 layers
   - Train only the custom classification layers
   - Use Adam optimizer with learning rate of 0.0001
   - Categorical cross-entropy loss function

2. **Fine-tuning Phase**:
   - Unfreeze the last few layers of the ResNet50 base
   - Train with a lower learning rate (0.00001)
   - Continue with the same optimizer and loss function

### Data Augmentation
To improve model generalization and prevent overfitting, the following augmentations are applied:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Shear transformation (±20%)
- Zoom (±20%)
- Horizontal flipping

## System Workflow

![System Workflow](workflow_diagram.png)

1. **Input**: A plant leaf image is provided to the system

2. **Preprocessing**:
   - Resize to 224×224 pixels (ResNet50 input size)
   - Convert to RGB format if needed
   - Normalize pixel values to [0,1]

3. **Prediction**:
   - The preprocessed image is fed to the trained ResNet50 model
   - The model outputs probability scores for each disease class
   - The class with the highest probability is selected as the prediction

4. **Report Generation**:
   - The system looks up information about the detected disease
   - A PDF report is generated containing:
     - The original image
     - Disease identification with confidence score
     - Disease description
     - Recommended treatment actions
     - Disclaimer for informational purposes

## Performance Metrics

The model typically achieves:
- **Accuracy**: 98%+ on the PlantVillage test set
- **Inference Time**: ~100ms per image on modern hardware
- **Model Size**: ~100MB

## Implementation Details

The model is implemented using TensorFlow and Keras, with the following key components:

1. **Data Loading**: `ImageDataGenerator` for efficient batch loading and augmentation
2. **Model Definition**: Functional API for flexible architecture design
3. **Training**: Two-phase approach with callbacks for early stopping and model checkpointing
4. **Prediction**: Efficient single-image processing
5. **Report Generation**: ReportLab library for PDF creation

## Extension and Customization

The model architecture can be customized in several ways:
- Different backbone networks (e.g., EfficientNet, DenseNet)
- Additional or modified augmentation techniques
- Alternative fine-tuning strategies
- Custom classification head architectures

The disease information database can be extended using the provided utility scripts to add new diseases and their remedies. 