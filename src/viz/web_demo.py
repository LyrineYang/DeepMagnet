import sys
import os
from pathlib import Path
import time

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import cv2  # Moved to top level

# Ensure project root is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import Backend Logic
from src.models.inference import load_model, run_inference
from src.data.physics import add_realistic_noise, GridConfig, synthesize_signal
from src.data.coil import CoilConfig, sample_trajectory

# --- Configuration & Styling ---
st.set_page_config(
    page_title="DeepMagnet: Interactive Paper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional, non-geeky look
st.markdown("""
<style>
    /* Import Inter font for that clean academic look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Center the main content to mimic a paper */
    .block-container {
        max-width: 900px;
        padding-top: 4rem;
        padding-bottom: 4rem;
        margin: 0 auto;
    }
    
    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -1px;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        color: #111827;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    h3 {
        font-weight: 600;
        font-size: 1.1rem;
        color: #374151;
        margin-top: 1.5rem;
    }
    p {
        line-height: 1.7;
        color: #374151;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .caption {
        text-align: center;
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Interactive Elements Styling */
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 9999px; /* Pill shape */
        height: 2.5em;
        padding: 0 1.5em;
        font-weight: 500;
        border: none;
        transition: transform 0.1s ease;
    }
    .stButton>button:hover {
        background-color: #333333;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards - Minimalist */
    .stMetric {
        background-color: #f9fafb; 
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #f3f4f6;
    }
    
    /* Canvas Container - Figure-like */
    .canvas-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin: 0 auto;
        display: block;
        box-shadow: 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Hide Streamlit Hamburger */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- State Management ---
@st.cache_resource
def get_model(ckpt_path, device):
    """Load model once and cache it."""
    # Check if file exists, if not allow fallback or error
    if not os.path.exists(ckpt_path):
        return None
    return load_model(ckpt_path, device=device)

@st.cache_resource
def get_configs():
    # H100 Config: 40cm box (-0.2 to 0.2)
    grid_cfg = GridConfig(size=(64, 64, 64), bounds=(-0.2, 0.2), voxel_size=0.4/64)
    
    # Coil Config: Line scan, matched to training
    coil_cfg = CoilConfig(
        radius=0.05,
        separation=0.02,
        turns=50,
        current=1.5,
        frequency=8000.0,
        sweep_steps=64,
        trajectory="line",
        samples=256,
        noise_std=0.01
    )
    return grid_cfg, coil_cfg

@st.cache_data
def get_grid_coords(_grid_cfg, device):
    D, H, W = _grid_cfg.size
    xs = torch.linspace(_grid_cfg.bounds[0], _grid_cfg.bounds[1], D)
    ys = torch.linspace(_grid_cfg.bounds[0], _grid_cfg.bounds[1], H)
    zs = torch.linspace(_grid_cfg.bounds[0], _grid_cfg.bounds[1], W)
    grid_coords = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
    return grid_coords.to(device)

def process_drawing(canvas_result, grid_cfg, coil_cfg, grid_coords, device):
    """
    Convert 2D drawing -> 3D Mask -> Clean Physics Signal
    """
    if canvas_result.image_data is None:
        return None, None
        
    # 1. Process Image: Resize 280x280 -> 64x64
    img_array = canvas_result.image_data[:, :, 0] # Take R channel (black/white)
    
    # Resize: cv2.resize expects (width, height) -> (W, H)
    # grid_cfg.size is (D, H, W). So we need (size[2], size[1])
    img_resized = cv2.resize(img_array, (grid_cfg.size[2], grid_cfg.size[1]))
    
    # 2. Extrude to 3D Mask (Imitate MNIST training data)
    # Training data: ~20% thickness, randomly placed. 
    # For demo: Place in center (Z=0) with 20% thickness for best visibility.
    binary_img = (img_resized > 10).astype(np.float32) # threshold
    D, H, W = grid_cfg.size
    mask_3d = np.zeros((D, H, W), dtype=np.float32)
    
    # Thickness: 20% of 64 ≈ 12 voxels
    thickness = int(D * 0.2) 
    z_center = D // 2
    z_start = max(0, z_center - thickness // 2)
    z_end = min(D, z_start + thickness)
    
    for z in range(z_start, z_end):
        mask_3d[z, :, :] = binary_img
        
    mask_tensor = torch.from_numpy(mask_3d).to(device)
    
    # 3. Physics Simulation
    # Use standard trajectory generator from training pipeline
    traj = sample_trajectory(coil_cfg, device)
    
    clean_signal = synthesize_signal(coil_cfg, traj, mask_tensor, grid_coords, noise_std=0.0)
    return clean_signal, mask_tensor

# --- UI Layout ---

def main():
    st.title("🧲 DeepMagnet Cloud")
    st.markdown("**Real-time Anti-Interference Electromagnetic Imaging Console**")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Control Panel")
        
        # Device Selection
        device_opt = st.selectbox("Compute Device", ["cpu", "mps", "cuda"] if torch.cuda.is_available() else ["cpu", "mps"])
        
        # Model Path
        default_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
        ckpt_path = st.text_input("Model Checkpoint Path", value=str(default_ckpt))
        
        st.divider()
        
        # Physics Parameters
        st.subheader("Environment")
        noise_level = st.slider("Soil Mineralization (Noise)", 0.0, 1.0, 0.0, 0.01, format="%d%%")
        
        st.divider()
        
        # Visualization Settings
        st.subheader("Visualization")
        iso_threshold = st.slider("Hologram Threshold", 0.1, 0.9, 0.5, 0.05, help="Sensitivity of the 3D reconstruction.")
        
        st.divider()
        st.success(f"System Ready • {device_opt.upper()}")

    # --- Loading Resources ---
    grid_cfg, coil_cfg = get_configs()
    # Handle model loading gracefully
    model = get_model(ckpt_path, device_opt)
    
    if model is None:
        st.error(f"Model not found at `{ckpt_path}`. Please train a model first or check the path.")
        st.stop()
        
    grid_coords = get_grid_coords(grid_cfg, device_opt)


    # --- Main Content (Academic Layout) ---
    st.title("DeepMagnet: Anti-Interference Electromagnetic Imaging")
    st.markdown("**Interactive Demonstration** • January 2026")
    
    st.markdown("""
    **Abstract.** Electromagnetic imaging allows for non-invasive detection of hidden metallic objects, yet it traditionally suffers from environmental interference. 
    Here, we present *DeepMagnet*, a foundation model approach that leverages domain randomization and large-scale synthetic data to robustly reconstruct 3D shapes from noisy, single-line scan signals.
    This interactive demo allows you to define an arbitrary target geometry and observe the model's reconstruction capability in real-time.
    """)
    
    st.divider()
    
    st.header("1. Method Verification")
    st.markdown("""
    To verify the model's zero-shot generalization capability, we invite you to draw an unseen 2D profile below. 
    This profile will be extruded to form a 3D target with **20% thickness** located at the physical center ($z=0$).
    The system simulates the magnetic response under a linear coil trajectory ($x \in [-0.08, 0.08]$m).
    """)
    
    # Center the canvas using columns
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # Canvas
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=20,
            stroke_color="white",
            background_color="#0f172a", 
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<p class="caption">Figure 1: Interactive Target Generation. Draw white shape on dark background.</p>', unsafe_allow_html=True)
        
        # Centered Action Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
             if st.button("Generate & Reconstruct", type="primary"):
                st.session_state['run_sim'] = True
    
    
    st.header("2. Experimental Results")
    
    if st.session_state.get('run_sim', False):
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
            # Progress Bar Flow
            progress_bar = st.progress(0, text="Initializing Physics Engine...")
            
            
            try:
                # Step 1: Physic Simulation
                progress_bar.progress(25, text="📡 Simulating Electromagnetic Field...")
                clean_signal, _ = process_drawing(canvas_result, grid_cfg, coil_cfg, grid_coords, device_opt)
                
                if clean_signal is not None:
                    # Step 2: Noise Injection
                    progress_bar.progress(50, text="📉 Injecting Environmental Noise...")
                    noisy_signal = add_realistic_noise(clean_signal, noise_level, drift_weight=0.3)
                    
                    # Step 3: Inference
                    progress_bar.progress(75, text="🧠 Running DeepONet Inference...")
                    pred_vol_tensor = run_inference(model, noisy_signal, grid_coords)
                    pred_vol = pred_vol_tensor.cpu().numpy()
                    
                    progress_bar.progress(100, text="✨ Rendering Hologram...")
                    time.sleep(0.3) # Small UI pause for effect
                    progress_bar.empty()
                    
                    # D. Visuals
                    
                    c_res1, c_res2 = st.columns([1.2, 1])
                    
                    with c_res1:
                        st.markdown("**Holographic Reconstruction**")
                        tab1, _ = st.tabs(["Isosurface View", "Slice View"]) # Slice view placeholder
                        
                        with tab1:
                            # Plotly Volume 3D -> Isosurface for better aesthetic
                            # Update coordinates to new physics bounds
                            bound = grid_cfg.bounds[1] # 0.2
                            X, Y, Z = np.mgrid[-bound:bound:64j, -bound:bound:64j, -bound:bound:64j]
                            
                            vol_fig = go.Figure(data=go.Isosurface(
                                x=X.flatten(),
                                y=Y.flatten(),
                                z=Z.flatten(),
                                value=pred_vol.flatten(),
                                isomin=iso_threshold,
                                isomax=1.0,
                                surface_count=5, 
                                colorscale='Spectral_r', # More scientific looking
                                opacity=0.4,
                                caps=dict(x_show=False, y_show=False, z_show=False)
                            ))
                            
                            # Add a core for higher probability
                            vol_fig.add_trace(go.Isosurface(
                                x=X.flatten(),
                                y=Y.flatten(),
                                z=Z.flatten(),
                                value=pred_vol.flatten(),
                                isomin=min(iso_threshold + 0.2, 0.95),
                                isomax=1.0,
                                surface_count=2,
                                colorscale='Spectral_r',
                                opacity=0.8, # Solid core
                                showscale=False
                            ))
                            
                            vol_fig.update_layout(
                                scene=dict(
                                    xaxis=dict(visible=False),
                                    yaxis=dict(visible=False),
                                    zaxis=dict(visible=False),
                                    bgcolor='rgba(0,0,0,0)',
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                                ),
                                margin=dict(l=0, r=0, b=0, t=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500,
                                showlegend=False
                            )
                            st.plotly_chart(vol_fig, use_container_width=True)
                        st.markdown(f'<p class="caption">Figure 2: 3D Reconstruction result. Iso-threshold: {iso_threshold}.</p>', unsafe_allow_html=True)

                    with c_res2:
                        st.markdown("**Sensor Signal Response**")
                        # Signal Heatmap
                        sig_np = noisy_signal.cpu().numpy() # (steps, samples)
                        heatmap_fig = go.Figure(data=go.Heatmap(
                            z=sig_np,
                            colorscale='Viridis',
                            showscale=False
                        ))
                        heatmap_fig.update_layout(
                            margin=dict(l=0, r=0, b=0, t=10),
                            xaxis_title="Time Sample",
                            yaxis_title="Trajectory Step",
                            height=300,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", size=10)
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        st.markdown('<p class="caption">Figure 3: Raw Voltage Signal (Simulated).</p>', unsafe_allow_html=True)
                        
                        # Metrics in a small table-like format
                    st.markdown("---")
                    m1, m2 = st.columns(2)
                    m1.metric("Confidence", f"{pred_vol.max():.1%}")
                    m2.metric("Signal Peak", f"{noisy_signal.abs().max():.2f} V")
                
                else:
                    progress_bar.empty()
                    st.warning("Please draw an object on the canvas first.")
            except Exception as e:
                progress_bar.empty()
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
