import streamlit as st
import numpy as np
from PIL import Image
from src.TerrainNavigator import Navigator

st.title("🛰️ Autonomous Terrain Relative Navigation System")
st.write("Upload a surface capture frame to evaluate landing safety zones in real-time.")

# 1. File Uploader UI
uploaded_file = st.file_uploader("Choose a terrain image (PPM, PNG, JPG)...", type=["ppm", "png", "jpg"])

if uploaded_file is not None:
    # Load image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Descent Frame", use_column_width=True)
    
    # 2. Add a slider for recruiters to play with parameters
    safety_radius = st.slider("Target Safety Buffer Radius (px)", min_value=10, max_value=100, value=40)
    
    with st.spinner("Processing neural pipeline and calculating spatial matrices..."):
        # 3. Call your existing backend
        # (Pass your initialized navigator or model components here)
        # best_point = navigator.locateDescentImageInReferenceImage(uploaded_file)
        
        st.success("Analysis Complete!")
        
        # 4. Display your generated visual outputs side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Suitability Heatmap")
            # st.image("outputs/latest/heatmap.png")
        with col2:
            st.subheader("3D Surface Reconstruction")
            # st.image("outputs/latest/terrain_3d.png")