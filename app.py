import pathlib
import platform

# Fix cross-platform path compatibility
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Steel Surface Defect Detection Using YOLOv5")
st.markdown("Upload an image of steel surface to detect defects automatically")

@st.cache_resource
def load_model():
    model_path = 'best.pt'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Load with force_reload to fix path issues
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=model_path,
                              force_reload=True,
                              trust_repo=True,
                              _verbose=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    model.conf = confidence_threshold
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a steel surface image for defect detection"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.subheader("📤 Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            with st.spinner('Detecting defects...'):
                results = model(image)
                detections = results.pandas().xyxy[0]
                
                annotated_img = np.squeeze(results.render())
                st.image(annotated_img, use_column_width=True)
        
        # Results summary
        st.markdown("---")
        st.subheader("📊 Detection Summary")
        
        if len(detections) > 0:
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Total Defects", len(detections))
            
            with col4:
                avg_confidence = detections['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            with col5:
                unique_defects = detections['name'].nunique()
                st.metric("Defect Types", unique_defects)
            
            # Results table
            st.subheader("📋 Detailed Results")
            results_df = detections[['name', 'confidence']].copy()
            results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
            results_df.columns = ['Defect Type', 'Confidence']
            st.dataframe(results_df, use_container_width=True)
            
            st.error("⚠️ Defects detected - REJECT")
        else:
            st.success("✅ No defects detected - ACCEPT")
    
    else:
        st.info("👆 Please upload a steel surface image to start detection")

if __name__ == "__main__":
    main()
