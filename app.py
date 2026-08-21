import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from scipy.signal import convolve2d
from streamlit_plotly_events import plotly_events

# ==========================================================
# PAGE SETUP & CUSTOM THEME
# ==========================================================
st.set_page_config(
    page_title="Spatial Matrix Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .panel-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .header-banner {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }
    .header-banner p {
        color: #e2e8f0;
        margin: 5px 0 0 0;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# APP HEADER
# ==========================================================
st.markdown("""
<div class="header-banner">
    <h1>🔬 Spatial Matrix & Kernel Explorer</h1>
    <p>Interactive Image Convolution & Edge Analysis Laboratory</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# CORE MATRIX DICTIONARY
# ==========================================================
@st.cache_data
def fetch_matrix_bank():
    return {
        "Pass-Through": np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
        "Box Blur": np.ones((3,3), dtype=float) / 9.0,
        "Gaussian Smoothing": np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float) / 16.0,
        "Edge Enhancement": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
        "Relief / Emboss": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float),
        "High-Pass Outline": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
        "Gradient X (Sobel)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
        "Gradient Y (Sobel)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
        "Laplacian Filter": np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    }

matrix_collection = fetch_matrix_bank()

# ==========================================================
# CONTROLS & SELECTION
# ==========================================================
col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    filter_category = st.selectbox(
        "Filter Category",
        ["Standard Filters", "Custom Matrix Builder"]
    )

with col_ctrl2:
    if filter_category == "Standard Filters":
        chosen_filter = st.selectbox(
            "Select Filter Preset",
            list(matrix_collection.keys())
        )
        active_matrix = matrix_collection[chosen_filter]
    else:
        chosen_filter = "Custom Matrix"
        st.write("Configure 3x3 Weights:")
        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            w00 = st.number_input("W11", value=0.0)
            w10 = st.number_input("W21", value=0.0)
            w20 = st.number_input("W31", value=0.0)
        with c_in2:
            w01 = st.number_input("W12", value=0.0)
            w11 = st.number_input("W22", value=1.0)
            w21 = st.number_input("W32", value=0.0)
        with c_in3:
            w02 = st.number_input("W13", value=0.0)
            w12 = st.number_input("W23", value=0.0)
            w22 = st.number_input("W33", value=0.0)
        active_matrix = np.array([[w00, w01, w02], [w10, w11, w12], [w20, w21, w22]], dtype=float)

# ==========================================================
# IMAGE HANDLER
# ==========================================================
def generate_synthetic_canvas():
    canvas = Image.new("L", (160, 120), 240)
    drawer = ImageDraw.Draw(canvas)
    drawer.rectangle([20, 20, 140, 100], fill=180)
    drawer.ellipse([50, 40, 110, 90], fill=100)
    drawer.rectangle([70, 60, 90, 80], fill=255)
    return np.array(canvas, dtype=float)

col_view_left, col_view_right = st.columns([1, 1])

with col_view_left:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.subheader("📁 Image Source")
    user_upload = st.file_uploader("Upload custom target image", type=["png", "jpg", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

if user_upload is not None:
    loaded_img = Image.open(user_upload).convert("L").resize((160, 120))
    target_grid = np.array(loaded_img, dtype=float)
    source_label = "User Uploaded Image"
else:
    target_grid = generate_synthetic_canvas()
    source_label = "Synthetic Reference Canvas"

# ==========================================================
# CONVOLUTION ENGINE
# ==========================================================
def execute_convolution(matrix_grid, kernel_weights):
    raw_output = convolve2d(matrix_grid, kernel_weights, mode="same", boundary="symm")
    normalized_output = np.abs(raw_output)
    if normalized_output.max() > normalized_output.min():
        normalized_output = ((normalized_output - normalized_output.min()) / (normalized_output.max() - normalized_output.min())) * 255.0
    return np.clip(normalized_output, 0, 255).astype(np.uint8)

processed_grid = execute_convolution(target_grid, active_matrix)

# ==========================================================
# DISPLAY VIEWPORTS
# ==========================================================
col_disp1, col_disp2 = st.columns(2)

with col_disp1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown(f"**Original Feed** ({source_label})")
    st.image(target_grid.astype(np.uint8), use_container_width=True, clamp=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_disp2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown(f"**Processed Result** ({chosen_filter})")
    st.image(processed_grid, use_container_width=True, clamp=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# INTERACTIVE INSPECTOR SESSION
# ==========================================================
if "row_index" not in st.session_state:
    st.session_state.row_index = target_grid.shape[0] // 2
if "col_index" not in st.session_state:
    st.session_state.col_index = target_grid.shape[1] // 2

r_pos = max(1, min(st.session_state.row_index, target_grid.shape[0] - 2))
c_pos = max(1, min(st.session_state.col_index, target_grid.shape[1] - 2))

st.markdown("### 🔍 Matrix Neighborhood Inspection")
col_map, col_calc = st.columns([1, 1.2])

with col_map:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.write("Hover over the heatmap to change sampling window:")
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=target_grid,
        colorscale="Viridis",
        showscale=False,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z}<extra></extra>"
    ))
    
    # Highlight neighborhood
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            fig_heatmap.add_shape(type="rect", x0=c_pos+dx-.5, x1=c_pos+dx+.5, y0=r_pos+dy-.5, y1=r_pos+dy+.5, line=dict(color="#f43f5e", width=1.5))

    fig_heatmap.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(autorange="reversed"))
    
    interaction_events = plotly_events(fig_heatmap, hover_event=True, click_event=False, override_height=350, key="heatmap_inspector")
    
    if interaction_events:
        last_event = interaction_events[-1]
        if "x" in last_event and "y" in last_event:
            st.session_state.col_index = max(1, min(int(round(last_event["x"])), target_grid.shape[1]-2))
            st.session_state.row_index = max(1, min(int(round(last_event["y"])), target_grid.shape[0]-2))
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_calc:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.subheader("Mathematical Breakdown")
    
    neighborhood_patch = target_grid[r_pos-1:r_pos+2, c_pos-1:c_pos+2]
    product_matrix = neighborhood_patch * active_matrix
    convolution_total = np.sum(product_matrix)
    
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        st.write("Neighborhood Region")
        st.dataframe(pd_df := np.round(neighborhood_patch, 1), use_container_width=True)
    with sub_c2:
        st.write("Element-wise Products")
        st.dataframe(np.round(product_matrix, 1), use_container_width=True)
        
    st.metric(label="Sum Accumulation Value", value=f"{convolution_total:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
