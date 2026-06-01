# demo.py

import streamlit as st
import os
import numpy as np
from PIL import Image
from src.TerrainNavigator import Navigator

# --- Page Configuration ---
st.set_page_config(
    page_title="Autonomous TRN System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Space/Tech Theme ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSlider > div > div > div > div { background-color: #ff4b4b; }
    div.stButton > button:first-child { background-color: #ff4b4b; color:white; }
    div.stButton > button:hover { background-color: #ff3333; color:white; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🛰️ Mission Configuration")
st.sidebar.markdown("Configure detection sensitivity and safety margins.")

conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.70, step=0.05)
safety_radius = st.sidebar.slider("Landing Buffer Radius (px)", 10, 100, 40, step=5)
reference_altitude = st.sidebar.number_input("Reference Altitude (m)", value=2000)

st.sidebar.markdown("---")
st.sidebar.caption("Autonomous Space Navigation Testbench")

# --- Main Interface Layout ---
st.title("🌌 Terrain Relative Navigation (TRN) System")
st.markdown("🧬 **Developed by:** [@Seagull28](https://github.com/Seagull28) | *Computer Vision & Space Systems Architecture*")
st.subheader("Autonomous Surface Hazard Detection & Optimal Landing Site Selection")

# --- System Overview Callout Panel ---
st.markdown("""
    <div style="background-color: #1e222b; padding: 20px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 25px;">
        <p style="margin: 0; font-size: 1.1rem; line-height: 1.6; color: #e2e8f0;">
            This system autonomously identifies safe spacecraft landing zones on planetary surfaces. 
            <strong>Upload a descent image ➔ YOLO detects craters ➔ a coarse-to-fine scoring algorithm evaluates 5 safety factors ➔ the optimal landing point is selected and visualised across a heatmap, 3D terrain map, and density overlay.</strong>
        </p>
        <p style="margin: 10px 0 0 0; font-size: 0.95rem;">
            🔗 <strong>Source Code & Architecture:</strong> 
            <a href="https://github.com/Seagull28/Terrain_Navigation_NN" target="_blank" style="color: #ff4b4b; text-decoration: none; font-weight: bold;">
                github.com/Seagull28/Terrain_Navigation_NN
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Interactive Sample Image Selector ---
st.markdown("### 📸 Select a Reference Telemetry Frame")

SAMPLES = {
    "Descent Frame A (Standard)": "data/TRN/Scene4.ppm",
    "Descent Frame B (High Density)": "data/TRN/Scene1.ppm",
    "Descent Frame C (Sparse Crater)": "data/TRN/Scene2.ppm"
}

if "selected_image" not in st.session_state:
    st.session_state.selected_image = SAMPLES["Descent Frame A (Standard)"]

cols = st.columns(len(SAMPLES))
for idx, (label, path) in enumerate(SAMPLES.items()):
    with cols[idx]:
        if os.path.exists(path):
            thumb = Image.open(path).resize((150, 150))
            st.image(thumb, caption=label, use_container_width=True)
            if st.button(f"Select Frame {chr(65+idx)}", key=f"btn_{idx}"):
                st.session_state.selected_image = path

st.markdown("---")

# --- Custom Image Upload Option ---
col_upload, col_current = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader("Or upload your own descent image (.ppm, .png, .jpg)", type=["ppm", "png", "jpg"])

if uploaded_file is not None:
    temp_path = os.path.join("data", "TRN", "temp_upload" + os.path.splitext(uploaded_file.name)[1])
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    target_image = temp_path
else:
    target_image = st.session_state.selected_image

with col_current:
    if os.path.exists(target_image):
        st.markdown("**Active Processing Target:**")
        st.image(Image.open(target_image).resize((120, 120)), caption=os.path.basename(target_image), use_container_width=True)

# --- Trigger Pipeline Execution ---
if st.button("🚀 Execute Autonomous Navigation Sequence", use_container_width=True):
    if not os.path.exists(target_image):
        st.error(f"Missing target telemetry data frame at `{target_image}`.")
    else:
        with st.spinner("Processing neural pipeline across tracking mesh matrices..."):
            
            # Defensive Exception Wrapping around backend pipeline processing
            try:
                navigator = Navigator(
                    referenceAltitude=reference_altitude,
                    referenceMap="ReferenceMap.ppm",
                    referenceCatalogue="catalogue",
                    datapath="data/TRN/"
                )
                navigator.detector.conf_threshold = conf_thresh
                
                # Execute pipeline and unpack computational map figures
                best_point, fig_heatmap, fig_density, fig_3d, im_localization = navigator.locateDescentImageInReferenceImage(target_image)
            except Exception as e:
                st.error(f"🚨 Navigation sequence failed: {e}")
                st.stop()
            
            # Cast coordinates safely to pure integers to avoid array formatting tags in UI
            safe_coords = (int(best_point[0]), int(best_point[1]))
            st.success(f"🎯 Target Acquired! Safe Landing Site Selected at Vector Matrix Coordinates: **{safe_coords}**")
            
            # --- Navigation Telemetry Summary ---
            st.markdown("### 📋 Navigation Telemetry Summary")
            
            # Initialize metrics to None to protect against silent reporting errors
            total_craters = None
            avg_conf = None
            landing_score = None
            min_dist = None

            # Dynamic run directory mapping strategy checks active session directories
            target_run_dir = None
            if hasattr(navigator, 'output_dir') and os.path.exists(navigator.output_dir):
                target_run_dir = navigator.output_dir
            elif os.path.exists("outputs"):
                # Fallback to scanning for the absolute newest timestamped execution subfolder
                all_runs = [os.path.join("outputs", d) for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]
                if all_runs:
                    target_run_dir = max(all_runs, key=os.path.getmtime)

            # Robust case-insensitive text processing logic
            if target_run_dir and os.path.exists(target_run_dir):
                log_path = os.path.join(target_run_dir, "report.txt")
                if os.path.exists(log_path):
                    with open(log_path, "r") as log_file:
                        for line in log_file:
                            line_lower = line.lower()
                            if "total craters detected:" in line_lower:
                                total_craters = int(line.split(":")[-1].strip())
                            elif "average confidence:" in line_lower:
                                avg_conf = float(line.split(":")[-1].strip())
                            elif "landing score:" in line_lower:
                                landing_score = float(line.split(":")[-1].strip())
                            elif "distance from nearest rim" in line_lower:
                                dist_str = line.split(":")[-1].lower().replace("px", "").strip()
                                min_dist = float(dist_str)

            # Generate KPI counters with structured fallback states
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric(label="Craters Detected", value=f"{total_craters}" if total_craters is not None else "N/A")
            with m_col2:
                st.metric(label="Avg Confidence", value=f"{avg_conf:.2f}" if avg_conf is not None else "N/A")
            with m_col3:
                st.metric(label="Best Landing Point", value=f"({safe_coords[0]}, {safe_coords[1]})")
            with m_col4:
                st.metric(label="Landing Safety Score", value=f"{landing_score:.2f}" if landing_score is not None else "N/A")
            with m_col5:
                st.metric(label="Nearest Hazard Distance", value=f"{int(min_dist)} px" if min_dist is not None else "N/A")

            st.markdown("---")
            
            # --- Visual Output Matrix Layout ---
            st.markdown("### 📊 Generated Telemetry Maps")
            
            # Layout tabs to separate high-dimension graphs and maintain vertical visibility
            tab_landing, tab_terrain, tab_density, tab_vectors = st.tabs([
                "🎯 Landing Heatmap & Overlay", 
                "⛰️ 3D Surface Reconstruction", 
                "🔴 Crater Density Footprint", 
                "📏 Crater Distance Vectors"
            ])
            
            with tab_landing:
                col_h, col_l = st.columns(2)
                with col_h:
                    st.markdown("### 🎯 Landing Suitability Heatmap")
                    st.pyplot(fig_heatmap)
                with col_l:
                    st.markdown("### 🔲 Object Localization Overlay")
                    st.image(im_localization, use_container_width=True)
                    
            with tab_terrain:
                # 1-4-1 layout centers plot tightly to block vertical scrolling on large screens
                _, center_col_3d, _ = st.columns([1, 4, 1])
                with center_col_3d:
                    st.markdown("### ⛰️ 3D Surface Reconstruction")
                    st.pyplot(fig_3d)
                    
            with tab_density:
                _, center_col_density, _ = st.columns([1, 4, 1])
                with center_col_density:
                    st.markdown("### 🔴 Crater Density Footprint")
                    st.pyplot(fig_density)
                
            with tab_vectors:
                _, center_col_vectors, _ = st.columns([1, 4, 1])
                with center_col_vectors:
                    st.markdown("### 📏 Crater Distance Vectors")
                    if target_run_dir and os.path.exists(target_run_dir):
                        saved_distance_path = os.path.join(target_run_dir, "distances.png")
                        if os.path.exists(saved_distance_path):
                            st.image(Image.open(saved_distance_path), use_container_width=True)
                        else:
                            st.warning("Distance vector map not generated.")
                    else:
                        st.warning("Distance vector map not generated.")
                
            # Perform directory file hygiene step
            if uploaded_file is not None and os.path.exists(temp_path):
                os.remove(temp_path)