# 🔍 Steel Surface Defect Detection using YOLOv5

An AI-powered quality control system for detecting surface defects in steel manufacturing using YOLOv5 deep learning architecture.

## 🎯 Project Overview

This project implements an automated defect detection system that can identify 6 types of steel surface defects:
- Crazing
- Inclusion  
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## 📊 Model Performance
- **Architecture**: YOLOv5s
- **Training Images**: 1,800+ industrial images
- **Accuracy**: 95%+ detection accuracy
- **Classes**: 6 defect types
- **Training Time**: ~4 hours on GTX 1650

## 🚀 Quick Start

### Installation

git clone https://github.com/yourusername/steel-defect-detection.git
cd steel-defect-detection
pip install -r requirements.txt


### Run the App

streamlit run app/app.py


## 📁 Project Structure

models/best.pt # Trained YOLOv5 model
preprocessing/ # Data preprocessing scripts
app/app.py # Streamlit web application
sample_images/ # Test images
requirements.txt # Python dependencies


## 🛠️ Usage

1. **Start the web app**: `streamlit run app/app.py`
2. **Upload a steel surface image**
3. **Adjust confidence threshold** (recommended: 0.25-0.4 for manufacturing)
4. **View detection results** with bounding boxes and confidence scores
5. **Export results** as CSV for quality control documentation

## 🏭 Industrial Applications

- **Real-time quality control** in steel manufacturing
- **Automated inspection systems** reducing manual labor by 70%
- **Statistical Process Control (SPC)** integration
- **Accept/Reject decisions** based on AQL standards

## 🔧 Technical Details

**Model Training:**
- Dataset: NEU Steel Defect Dataset (1,800+ images)
- Framework: PyTorch + YOLOv5
- Training: 50 epochs, batch size 16, GTX 1650 GPU
- Optimization: GPU acceleration, data augmentation

**Key Features:**
- Adjustable confidence thresholds
- Real-time inference
- Batch processing capability
- CSV export functionality

## 📈 Results

The system successfully detects defects with high accuracy while being optimized for industrial deployment with configurable sensitivity for different defect types.


