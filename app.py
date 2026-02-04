import streamlit as st
import os
import tempfile
import time
import base64
import numpy as np
import torch
import nibabel as nib
import plotly.graph_objects as go
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
)
from monai.data import DataLoader, Dataset

# Set Page Config
st.set_page_config(
    page_title="NeuroScan 3D: AI Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for "Wow" Factor ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif; 
            background-color: #0e1117;
            color: #fafafa;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        }

        /* Card Styling */
        .card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        h1, h2, h3 {
            font-weight: 700 !important;
            background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            border-radius: 8px;
            transition-duration: 0.4s;
            font-weight: 600;
        }

        .stButton>button:hover {
            box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
            transform: translateY(-2px);
        }

        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            color: #38bdf8;
            font-weight: 600;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #94a3b8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.title("🧠 NeuroScan 3D")
st.markdown("### Advanced AI Brain Tumor Analysis Dashboard")
st.markdown("---")

# --- Sidebar: Configuration & Uploads ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_path = st.text_input(
        "Model Checkpoint Path (.pt)",
        value=r"C:\Users\moham\OneDrive\Desktop\deployment hackathon\model_best.pt",
        help=r"C:\Users\moham\OneDrive\Desktop\deployment hackathon\model_best.pt"
    )
    
    st.markdown("---")
    st.header("📂 Upload MRI Scans")
    st.info("Please upload all 4 sequences.")
    
    # File uploaders
    flair_file = st.file_uploader("FLAIR Image", type=["nii", "nii.gz"], key="flair")
    t1_file = st.file_uploader("T1 Image", type=["nii", "nii.gz"], key="t1")
    t1ce_file = st.file_uploader("T1ce Image", type=["nii", "nii.gz"], key="t1ce")
    t2_file = st.file_uploader("T2 Image", type=["nii", "nii.gz"], key="t2")
    
    run_btn = st.button("🚀 Run Analysis", type="primary")

# --- Persistence (Session State) ---
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "vol_metrics" not in st.session_state:
    st.session_state.vol_metrics = {}
if "feature_maps" not in st.session_state:
    st.session_state.feature_maps = {}

# --- Helper Functions ---

@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            # print(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            return None
    else:
        st.warning(f"Checkpoint not found at {checkpoint_path}. Using initialized model (random weights).")
    
    model.eval()
    return model

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    return None

def normalize_image_for_display(img_data):
    """Normalize image to 0-1 range for visualization."""
    return (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8)

# --- Main Logic ---

if run_btn:
    st.session_state.analysis_complete = False # Reset
    if not (flair_file and t1_file and t1ce_file and t2_file):
        st.error("❌ Please upload all 4 MRI files (FLAIR, T1, T1CE, T2) to proceed.")
    else:
        # Load Model
        with st.spinner("Loading Model..."):
            model = load_model(model_path)
            
        if model:
            # Save files temporarily for MONAI loaders
            files = {
                "flair": save_uploaded_file(flair_file),
                "t1": save_uploaded_file(t1_file),
                "t1ce": save_uploaded_file(t1ce_file),
                "t2": save_uploaded_file(t2_file)
            }
            
            # Preprocessing Pipeline
            input_paths = [files["t1ce"], files["t1"], files["t2"], files["flair"]] # Order requested: T1CE, T1, T2, FLAIR
            
            # Manual loading and stacking
            st.toast("Preprocessing images...", icon="🔄")
            
            try:
                # Load images
                imgs = []
                for p in input_paths:
                    nii = nib.load(p)
                    imgs.append(nii.get_fdata())
                
                # Stack: (H, W, D, 4) -> (4, H, W, D) for ChannelFirst
                img_stack = np.stack(imgs, axis=0) 
                
                # Create wrapper for transforms
                t1ce_nii = nib.load(files["t1ce"])
                original_affine = t1ce_nii.affine
                
                combined_img = nib.Nifti1Image(np.moveaxis(img_stack, 0, -1), original_affine) # (H,W,D,4)
                
                combined_path = os.path.join(tempfile.gettempdir(), "combined_4d.nii.gz")
                nib.save(combined_img, combined_path)
                
                val_data = [{"image": combined_path}]
                
                val_transforms = Compose([
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
                    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                ])
                
                val_ds = Dataset(data=val_data, transform=val_transforms)
                val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
                
                # Inference
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        inputs = batch_data["image"].to(device)
                        
                        st.toast("Running AI Inference...", icon="🧠")
                        start_time = time.time()
                        
                        roi_size = (128, 128, 128)
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(
                            inputs, roi_size, sw_batch_size, model
                        )
                        
                        probs = torch.sigmoid(val_outputs)
                        seg = (probs > 0.5).float()
                        seg_np = seg.cpu().numpy()[0] # (3, H, W, D)
                        
                        # Construct label map
                        label_map = np.zeros_like(seg_np[0])
                        label_map[seg_np[1] == 1] = 2 # Edema
                        label_map[seg_np[0] == 1] = 1 # Necrotic
                        label_map[seg_np[2] == 1] = 3 # Enhancing
                        
                        inference_time = time.time() - start_time
                        
                        # Store in Session State
                        st.session_state.feature_maps = {
                            "label_map": label_map,
                            "flair": batch_data["image"][0][3].cpu().numpy(), # Original FLAIR
                            "inference_time": inference_time
                        }
                        
                        # Volumetrics
                        voxel_volume = 1.0
                        vol_necrotic = np.sum(label_map == 1) * voxel_volume / 1000
                        vol_edema = np.sum(label_map == 2) * voxel_volume / 1000
                        vol_enhancing = np.sum(label_map == 3) * voxel_volume / 1000
                        total_vol = vol_necrotic + vol_edema + vol_enhancing
                        
                        st.session_state.vol_metrics = {
                            "necrotic": vol_necrotic,
                            "edema": vol_edema,
                            "enhancing": vol_enhancing,
                            "total": total_vol
                        }
                        
                        st.session_state.analysis_complete = True
                        
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# --- Results Display (Persistent Loop) ---
if st.session_state.analysis_complete:
    label_map = st.session_state.feature_maps["label_map"]
    flair_raw = st.session_state.feature_maps["flair"]
    metrics = st.session_state.vol_metrics
    
    st.success(f"✅ Analysis Complete in {st.session_state.feature_maps['inference_time']:.2f}s")
    
    # --- Feature 3: Automated Volumetrics Report ---
    st.markdown("### 📊 Automated Volumetrics Report")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Necrotic Core (Red)", f"{metrics['necrotic']:.2f} cm³", delta_color="normal")
    c2.metric("Peritumoral Edema (Green)", f"{metrics['edema']:.2f} cm³", delta_color="normal")
    c3.metric("Enhancing Tumor (Yellow)", f"{metrics['enhancing']:.2f} cm³", delta_color="normal")
    c4.metric("Total Tumor Volume", f"{metrics['total']:.2f} cm³")
    
    st.markdown("---")
    
    # --- Feature 1: Slice Explorer ---
    st.markdown("### 👁️ Slice Explorer")
    
    flair_vol = normalize_image_for_display(flair_raw)
    
    max_slice = flair_vol.shape[2] - 1
    slice_idx = st.slider("Select Slice Depth (Z-axis)", 0, max_slice, max_slice // 2)
    
    col_view1, col_view2 = st.columns(2)
    
    with col_view1:
        st.caption("MRI (FLAIR Sequence)")
        fig_mri = go.Figure(data=go.Heatmap(
            z=np.rot90(flair_vol[:, :, slice_idx]), 
            colorscale='Gray', 
            showscale=False
        ))
        fig_mri.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_mri, use_container_width=True)

    with col_view2:
        st.caption("AI Segmentation Overlay")
        seg_slice = np.rot90(label_map[:, :, slice_idx])
        
        fig_seg_only = go.Figure(data=go.Heatmap(
            z=seg_slice,
            colorscale=[
                [0.0, 'black'],
                [0.25, 'black'],
                [0.25, 'red'],      # Necrotic
                [0.5, 'red'],
                [0.5, 'green'],     # Edema
                [0.75, 'green'],
                [0.75, 'yellow'],   # Enhancing
                [1.0, 'yellow']
            ],
            showscale=False,
            zmin=0, zmax=3
        ))
        
        fig_seg_only.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_seg_only, use_container_width=True)
    
    st.markdown("---")

    # --- Feature 2: 3D Tumor Reconstruction ---
    st.markdown("### 🔮 3D Tumor Reconstruction")
    st.caption("Interactive 3D model. Drag to rotate, scroll to zoom.")
    
    # Downsample for performance (reduces size by 8x)
    step = 2
    vol_downsampled = label_map[::step, ::step, ::step]
    
    X, Y, Z = np.mgrid[0:vol_downsampled.shape[0], 0:vol_downsampled.shape[1], 0:vol_downsampled.shape[2]]
    
    fig_3d = go.Figure()
    
    # Necrotic (1) = Red
    if np.any(vol_downsampled == 1):
        fig_3d.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=(vol_downsampled == 1).flatten().astype(int),
            isomin=0.5, isomax=1.0,
            colorscale=[[0, 'red'], [1, 'red']], opacity=0.3, surface_count=1,
            name='Necrotic', showscale=False
        ))
        
    # Edema (2) = Green
    if np.any(vol_downsampled == 2):
        fig_3d.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=(vol_downsampled == 2).flatten().astype(int),
            isomin=0.5, isomax=1.0,
            colorscale=[[0, 'green'], [1, 'green']], opacity=0.2, surface_count=1,
            name='Edema', showscale=False
        ))
        
    # Enhancing (3) = Yellow
    if np.any(vol_downsampled == 3):
        fig_3d.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=(vol_downsampled == 3).flatten().astype(int),
            isomin=0.5, isomax=1.0,
            colorscale=[[0, 'yellow'], [1, 'yellow']], opacity=0.6, surface_count=1,
            name='Enhancing', showscale=False
        ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    
    # --- Feature 4: PDF Report Generation ---
    from fpdf import FPDF
    from datetime import datetime

    class PDF(FPDF):
        def header(self):
            # Logo or Medical Header
            self.set_font('Arial', 'B', 20)
            self.set_text_color(25, 25, 112) # Midnight Blue
            self.cell(0, 10, 'NeuroScan 3D', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Advanced AI Brain Tumor Analysis', 0, 1, 'C')
            self.ln(5)
            # Line break
            self.set_draw_color(0, 0, 0)
            self.set_line_width(0.5)
            self.line(10, 30, 200, 30)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()} - Generated by NeuroScan 3D AI', 0, 0, 'C')

    def create_pdf(metrics):
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Patient / Scan Info
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 10, txt="Date of Analysis:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(230, 230, 250) # Lavender background
        pdf.cell(0, 10, txt="  Volumetric Analysis Report", ln=True, fill=True)
        pdf.ln(10)
        
        # Results Table
        pdf.set_font("Arial", 'B', 12)
        
        # Header Row
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(100, 10, txt=" Tumor Region", border=1, fill=True)
        pdf.cell(60, 10, txt=" Volume (cm3)", border=1, fill=True, ln=True)
        
        # Data Rows
        pdf.set_font("Arial", '', 12)
        
        pdf.cell(100, 10, txt=" Necrotic Core (Red)", border=1)
        pdf.cell(60, 10, txt=f" {metrics['necrotic']:.2f}", border=1, ln=True)
        
        pdf.cell(100, 10, txt=" Peritumoral Edema (Green)", border=1)
        pdf.cell(60, 10, txt=f" {metrics['edema']:.2f}", border=1, ln=True)
        
        pdf.cell(100, 10, txt=" Enhancing Tumor (Yellow)", border=1)
        pdf.cell(60, 10, txt=f" {metrics['enhancing']:.2f}", border=1, ln=True)
        
        # Total
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(100, 10, txt=" Total Tumor Volume", border=1, fill=True)
        pdf.cell(60, 10, txt=f" {metrics['total']:.2f}", border=1, fill=True, ln=True)
        
        pdf.ln(20)
        
        # Interpretation / Warning
        pdf.set_text_color(200, 0, 0) # Red
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, txt="IMPORTANT INTERPRETATION NOTICE:", ln=True)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, txt="This report is automatically generated by an experimental AI algorithm. The segmentation and volumetric calculations have not been verified by a radiologist. This document should NOT be used for direct clinical diagnosis or treatment planning without expert medical review.")
        
        # Return byte string
        return pdf.output(dest='S').encode('latin-1')

    # Generate the PDF in memory
    pdf_data = create_pdf(metrics)
    
    st.markdown("### 📄 Export Results")
    st.download_button(
        label="📥 Download Professional PDF Report",
        data=pdf_data,
        file_name=f"NeuroScan_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        key="download_pdf"
    )

else:
    st.info("👋 Upload sequences and define model path to start.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 12px;'>NeuroScan 3D v1.0 | Powered by MONAI & Streamlit</div>", unsafe_allow_html=True)
