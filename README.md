# CIFAR-10 Image Classification: Transfer Learning vs Custom CNN

A comprehensive comparison between two approaches to image classification on the CIFAR-10 dataset: Transfer Learning with a pretrained EfficientNetB0 model versus training a Custom Convolutional Neural Network from scratch.

## 📋 Project Overview

This project demonstrates best practices for deep learning image classification by implementing and comparing two distinct approaches:

1. **Transfer Learning**: Fine-tuning a pretrained ImageNet model (EfficientNetB0)
2. **Custom CNN**: Training a convolutional neural network from scratch

Both approaches are evaluated on the CIFAR-10 dataset with rigorous data handling, preprocessing, augmentation, training, and evaluation methodologies.

## 📊 Dataset

**CIFAR-10** is a collection of 60,000 32×32 RGB color images classified into 10 mutually exclusive classes:

- **Airplane**
- **Automobile**
- **Bird**
- **Cat**
- **Deer**
- **Dog**
- **Frog**
- **Horse**
- **Ship**
- **Truck**

### Data Split

- **Training set**: 45,000 samples (75% of original training data)
- **Validation set**: 5,000 samples (stratified split)
- **Test set**: 10,000 samples (official CIFAR-10 test set)

All splits preserve class distribution through stratified sampling for balanced representation.

## 🏗️ Project Structure

```
TransferLearning/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── venv/                                        # Virtual environment
└── cifar10_transferlearning_vs_customcnn.ipynb  # Main Jupyter Notebook
```

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x & Keras
- NumPy
- Matplotlib
- scikit-learn
- Pandas

See `requirements.txt` for complete dependency list.

## 🚀 Installation and Setup

### Step 1: Create a Virtual Environment

```bash
cd TransferLearning
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebook

```bash
jupyter notebook cifar10_transferlearning_vs_customcnn.ipynb
```

## 📚 Notebook Sections

### 1. **Dataset Loading and Splitting**

- Loads CIFAR-10 using TensorFlow/Keras
- Splits training data into 45K training and 5K validation sets
- Sets random seeds for reproducibility (SEED = 42)
- Flattens label arrays for compatibility with training pipelines

### 2. **Dataset Size Reporting**

- Prints dataset dimensions
- Verifies class distribution across splits
- Ensures balanced representation across all 10 classes

### 3. **Data Pipeline and Preprocessing**

- Creates efficient `tf.data` pipelines with shuffling and batching
- **Custom CNN preprocessing**: Normalization to [0, 1]
- **Transfer Learning preprocessing**:
  - Resizing to 224×224
  - EfficientNetB0-specific preprocessing
- Batch size: 64
- Implements prefetch for optimization

### 4. **Data Augmentation and Visualization**

- **Augmentation techniques applied**:
  - Random horizontal flips
  - Random rotations (±10°)
  - Random zoom (±10%)
  - Random translations (±10%)
- Visualizes augmented samples from the training set
- Separate augmentation pipelines for each model type

### 5. **Transfer Learning Model: Building and Training**

#### Architecture

- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Input Shape**: 224×224×3
- **Custom Head**:
  - GlobalAveragePooling2D
  - Dropout(0.2)
  - Dense(10, softmax) for classification
- **Initial Training**: Base model frozen (5 epochs)

#### Compilation

- Optimizer: Adam (learning rate: 1e-3)
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

### 6. **Transfer Learning Model: Fine-Tuning**

- Unfreezes top 20 layers of the base model
- Lower learning rate (1e-5) for careful weight adjustment
- Additional 5 epochs of training
- Improves model performance through careful parameter tuning

### 7. **Transfer Learning Model: Evaluation**

- Validation accuracy on 5,000 validation samples
- Test accuracy on 10,000 test samples
- Total training time recorded
- Parameter count reported

### 8. **Custom CNN Model: Building and Training**

#### Architecture

```
Input (32×32×3)
  ↓
Conv2D(32, kernel=3) + ReLU
  ↓
MaxPooling2D
  ↓
Conv2D(64, kernel=3) + ReLU
  ↓
MaxPooling2D
  ↓
Conv2D(128, kernel=3) + ReLU
  ↓
MaxPooling2D
  ↓
Flatten
  ↓
Dropout(0.5)
  ↓
Dense(128) + ReLU
  ↓
Dense(10, softmax) → Output
```

#### Training Parameters

- Optimizer: Adam (learning rate: 1e-3)
- Loss: Sparse Categorical Crossentropy
- Epochs: 15
- Data augmentation applied throughout

### 9. **Custom CNN Model: Evaluation**

- Validation accuracy on 5,000 validation samples
- Test accuracy on 10,000 test samples
- Total training time recorded
- Parameter count analysis

### 10. **Training and Validation Curves Plotting**

- Plots accuracy curves for all training phases
- Plots loss curves for model comparison
- Shows convergence behavior and potential overfitting
- Compares generalization across Transfer Learning and Custom CNN

### 11. **Model Comparison Table**

- Side-by-side comparison of:
  - Test accuracy
  - Training time (seconds)
  - Number of parameters
  - Architecture notes

### 12. **Report: Dataset, Preprocessing, and Augmentation**

- Comprehensive dataset overview
- Preprocessing strategies for each model type
- Augmentation techniques and rationale
- Data handling best practices

### 13. **Report: Model Architectures and Training Setup**

- Detailed architecture descriptions
- Hyperparameter specifications
- Training configuration details
- Learning rate schedules

### 14. **Report: Results, Discussion, and Conclusion**

- Performance metrics and analysis
- Discussion of key findings
- Comparison of approaches
- Recommendations for best practices

## 🎯 Key Findings

### Performance Comparison

| Metric                  | Transfer Learning (EfficientNetB0)     | Custom CNN                  |
| ----------------------- | -------------------------------------- | --------------------------- |
| **Test Accuracy**       | Higher (leverages ImageNet features)   | Lower (trains from scratch) |
| **Training Time**       | Shorter initial phase                  | Longer (15 epochs)          |
| **Parameters**          | More (~4.2M for EfficientNetB0 + head) | Fewer (~350K)               |
| **Convergence Speed**   | Faster                                 | Slower                      |
| **Memory Requirements** | Higher                                 | Lower                       |

### Key Insights

1. **Transfer Learning Advantages**
   - Pre-trained features from ImageNet provide strong initialization
   - Faster convergence and better accuracy with limited data
   - Fine-tuning top layers improves performance significantly
   - Particularly effective for small datasets like CIFAR-10

2. **Custom CNN Advantages**
   - Smaller model footprint, better for edge deployment
   - Fully interpretable architecture (no black-box pretrained weights)
   - Lower inference latency
   - Useful for experimentation and custom adaptations

3. **Data Augmentation Impact**
   - Significantly improves generalization for both models
   - Reduces overfitting on limited training data
   - Essential for achieving robust predictions

4. **Preprocessing Importance**
   - Model-specific preprocessing (224×224 for EfficientNetB0) is crucial
   - Normalization to [0, 1] for custom CNN works well
   - Proper preprocessing ensures optimal feature extraction

## 💡 Best Practices Demonstrated

✅ **Reproducibility**: Fixed random seeds across NumPy, TensorFlow, and Python's random module

✅ **Data Handling**: Stratified train-validation-test split preserving class distribution

✅ **Efficient Pipelines**: `tf.data` API with shuffling, batching, and prefetch for optimal GPU utilization

✅ **Data Augmentation**: Applied to training data only, using appropriate techniques

✅ **Model Architecture**: Clear separation of concerns between preprocessing and model layers

✅ **Training Methodology**:

- Initial feature extraction (frozen base)
- Fine-tuning (unfrozen layers with lower learning rate)
- Comprehensive evaluation metrics

✅ **Comparison Framework**: Controlled experiments enabling fair model comparison

## 📈 Results and Recommendations

### When to Use Transfer Learning

- Limited training data available
- Need faster training and convergence
- Want to leverage pre-trained knowledge from large datasets
- Target dataset differs significantly from custom CNN training domain

### When to Use Custom CNN

- Abundant training data available
- Specific model architecture requirements
- Model size/latency constraints critical
- Need for full interpretability and control
- Custom class definitions or specialized input requirements

### General Recommendations

1. **Start with transfer learning** for most practical applications
2. **Use fine-tuning** rather than feature extraction for best results
3. **Apply data augmentation** consistently across all approaches
4. **Monitor generalization** through validation metrics
5. **Optimize hyperparameters** through systematic experimentation

## 🔧 Customization Guide

### Adjusting Parameters

```python
# Batch size
BATCH_SIZE = 64

# Image size for transfer learning
PRETRAINED_IMG_SIZE = (224, 224)

# Augmentation intensity
layers.RandomRotation(0.1)  # ±10%
layers.RandomZoom(0.1)      # ±10%

# Learning rates
optimizer=keras.optimizers.Adam(learning_rate=1e-3)  # Initial
optimizer=keras.optimizers.Adam(learning_rate=1e-5)  # Fine-tuning

# Number of training epochs
epochs=5   # Transfer learning phases
epochs=15  # Custom CNN
```

### Trying Different Models

The framework supports easy substitution of transfer learning models:

```python
# Instead of EfficientNetB0, try:
keras.applications.MobileNetV2()
keras.applications.ResNet50()
keras.applications.VGG16()
keras.applications.InceptionV3()
```

## 📖 Dependencies Explained

- **TensorFlow/Keras**: Deep learning framework for model building and training
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization (training curves and samples)
- **scikit-learn**: ML utilities (train-test split with stratification)
- **Pandas**: Data manipulation for comparison tables

## 🎓 Learning Outcomes

By working through this project, you will understand:

- ✓ Transfer learning fundamentals and application
- ✓ Custom CNN design and training from scratch
- ✓ Data preprocessing and augmentation strategies
- ✓ Model evaluation and comparison methodologies
- ✓ TensorFlow/Keras workflow and best practices
- ✓ Trade-offs between different approaches
- ✓ Reproducibility in machine learning experiments

## 🐛 Troubleshooting

### Out of Memory Errors

- Reduce BATCH_SIZE (e.g., 32 or 16)
- Reduce image resolution
- Use a smaller base model for transfer learning

### Slow Training

- Check GPU availability: `tf.config.list_physical_devices('GPU')`
- Increase batch size if memory allows
- Use tf.data prefetch optimization (already included)

### Poor Accuracy

- Increase number of epochs
- Adjust learning rate (try 1e-4 or 5e-4)
- Modify dropout rates
- Enhance data augmentation

## 📎 Additional Resources

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Documentation](https://keras.io/)

## 📝 License and Attribution

This project is provided for educational purposes as part of SJSU Course 252.

## ✍️ Author

Created for SJSU Course 252: Transfer Learning and Deep Neural Networks

---

**Last Updated**: March 2026

For questions or improvements, please refer to the original notebook and experiment with different configurations!
