import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from scipy.signal import convolve2d
from streamlit_plotly_events import plotly_events
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionLab | Image Processing Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODERN EMERALD & SLATE THEME CSS
# ============================================================

st.markdown("""
<style>

* {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.stApp {
    background:
    radial-gradient(circle at 5% 5%, #f0fdf4 0%, transparent 20%),
    radial-gradient(circle at 95% 5%, #ecfdf5 0%, transparent 20%),
    linear-gradient(135deg, #f8fafc, #f1f5f9);
}

.block-container {
    padding: 10px 24px 20px 24px !important;
    max-width: 1500px !important;
}

header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #0d9488;
    margin: 4px 0 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.title-line {
    width: 140px;
    height: 4px;
    margin: 4px auto 16px;
    background: linear-gradient(
        90deg,
        #0d9488,
        #14b8a6,
        #0d9488
    );
    border-radius: 2px;
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {
    font-size: 17px;
    font-weight: 700;
    color: white;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    background: linear-gradient(
        90deg,
        #0f766e,
        #0d9488,
        #14b8a6
    );
    margin: 12px 0 8px;
    letter-spacing: 0.3px;
}


/* ============================================================
   THEORY
   ============================================================ */

.theory-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    width: 100%;
}

.theory-box {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px;
    min-height: 150px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.03);
}

.theory-head {
    display: flex;
    align-items: center;
    gap: 8px;
}

.theory-number {
    display: inline-flex;
    width: 28px;
    height: 28px;
    min-width: 28px;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    color: white;
    font-size: 12px;
    font-weight: bold;
}

.emerald { background: linear-gradient(145deg, #0d9488, #0f766e); }
.teal { background: linear-gradient(145deg, #14b8a6, #0d9488); }
.cyan { background: linear-gradient(145deg, #06b6d4, #0891b2); }
.slate { background: linear-gradient(145deg, #475569, #334155); }

.theory-heading {
    font-size: 14px;
    color: #0f766e;
    font-weight: 700;
}

.theory-text {
    font-size: 13.5px;
    line-height: 1.5;
    color: #334155;
    margin-top: 8px;
    min-height: 60px;
}

.formula-label {
    font-size: 11px;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 3px;
    text-transform: uppercase;
}

.formula-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    padding: 6px;
    font-size: 12px;
    font-weight: 700;
    color: #0f172a;
}


/* ============================================================
   CARDS & SELECTION
   ============================================================ */

.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.03);
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #0f766e;
    margin-bottom: 6px;
}

.selection-card, .selection-card-blue {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #99f6e4;
    border-radius: 10px;
    padding: 10px 14px;
    min-height: 85px;
}

.selection-heading {
    color: #0f766e;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
}


/* ============================================================
   INFO BOXES & FOOTER
   ============================================================ */

.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
}

.info-title {
    font-size: 11px;
    color: #64748b;
    font-weight: 600;
}

.info-value {
    font-size: 13px;
    font-weight: 700;
    color: #0f766e;
}

.small-note {
    color: #64748b;
    font-size: 11.5px;
    text-align: center;
    margin-top: 4px;
}

.footer {
    text-align: center;
    background: linear-gradient(90deg, #ccfbf1, #e0f2fe);
    padding: 10px;
    margin-top: 15px;
    color: #0f766e;
    font-size: 12px;
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid #99f6e4;
}


/* ============================================================
   STREAMLIT OVERRIDES
   ============================================================ */

div[data-testid="stVerticalBlock"] { gap: .35rem; }
div[data-testid="stHorizontalBlock"] { gap: .75rem; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="main-title">
    ⚡ VISIONLAB STUDIO
</div>
<div class="subtitle">
    Advanced Spatial Filtering & Edge Intelligence Suite
</div>
<div class="title-line"></div>
""", unsafe_allow_html=True)


# ============================================================
# THEORY SECTION
# ============================================================

st.markdown('<div class="section-title">📚 Fundamental Concepts</div>', unsafe_allow_html=True)

st.markdown("""
<div class="theory-grid">
    <div class="theory-box">
        <div class="theory-head">
            <span class="theory-number emerald">01</span>
            <span class="theory-heading">Spatial Domain Processing</span>
        </div>
        <div class="theory-text">
            Directly modifies pixel intensity values across a neighborhood matrix (kernel) to achieve smoothing, sharpening, or feature extraction.
        </div>
        <div class="formula-label">Core Operation:</div>
        <div class="formula-box">g(x,y) = ΣΣ f(x+i,y+j) × h(i,j)</div>
    </div>
    <div class="theory-box">
        <div class="theory-head">
            <span class="theory-number teal">02</span>
            <span class="theory-heading">Image Filtering</span>
        </div>
        <div class="theory-text">
            Applies low-pass or high-pass frequency alterations. Averaging filters reduce high-frequency noise while preserving baseline structure.
        </div>
        <div class="formula-label">Averaging Mask:</div>
        <div class="formula-box">h(i,j) = 1/9 (for standard 3×3)</div>
    </div>
    <div class="theory-box">
        <div class="theory-head">
            <span class="theory-number cyan">03</span>
            <span class="theory-heading">Gradient Operators</span>
        </div>
        <div class="theory-text">
            Measures directional intensity rate-of-change to detect object boundaries and edge contours efficiently (Sobel, Prewitt, Laplacian).
        </div>
        <div class="formula-label">Magnitude Formula:</div>
        <div class="formula-box">G = √(Gx² + Gy²)</div>
    </div>
    <div class="theory-box">
        <div class="theory-head">
            <span class="theory-number slate">04</span>
            <span class="theory-heading">Matrix Convolution</span>
        </div>
        <div class="theory-text">
            Slides the kernel across every image pixel, computing element-wise dot products to synthesize the new filtered coordinate value.
        </div>
        <div class="formula-label">Transformation:</div>
        <div class="formula-box">Neighborhood Matrix * Kernel → Pixel</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# OPERATION SELECTION
# ============================================================

st.markdown('<div class="section-title">⚙️ Processing Configuration</div>', unsafe_allow_html=True)

select_col1, select_col2 = st.columns(2)

with select_col1:
    st.markdown('<div class="selection-card"><div class="selection-heading">📂 Category</div></div>', unsafe_allow_html=True)
    operation_type = st.selectbox("Category", ["Spatial Domain Methods", "Gradient Operators"], label_visibility="collapsed")

with select_col2:
    if operation_type == "Spatial Domain Methods":
        st.markdown('<div class="selection-card-blue"><div class="selection-heading">🎯 Spatial Operation</div></div>', unsafe_allow_html=True)
        operation = st.selectbox("Spatial Operation", ["Identity", "Blur / Average", "Gaussian-like Blur", "Sharpen", "Emboss", "Outline", "Threshold", "Invert", "Custom Kernel"], label_visibility="collapsed")
    else:
        st.markdown('<div class="selection-card-blue"><div class="selection-heading">🎯 Gradient Operator</div></div>', unsafe_allow_html=True)
        operation = st.selectbox("Gradient Operator", ["Sobel X", "Sobel Y", "Sobel Magnitude", "Prewitt X", "Prewitt Y", "Laplacian", "Roberts X", "Roberts Y"], label_visibility="collapsed")


# ============================================================
# KERNEL DEFINITIONS
# ============================================================

def get_kernel(op):
    kernels = {
        "Identity": np.array([[0,0,0],[0,1,0],[0,0,0]], float),
        "Blur / Average": np.ones((3,3), float) / 9,
        "Gaussian-like Blur": np.array([[1,2,1],[2,4,2],[1,2,1]], float) / 16,
        "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], float),
        "Emboss": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], float),
        "Outline": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], float),
        "Sobel X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),
        "Sobel Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float),
        "Prewitt X": np.array([[-1,0,1],[-1,0,1],[-1,0,1]], float),
        "Prewitt Y": np.array([[-1,-1,-1],[0,0,0],[1,1,1]], float),
        "Laplacian": np.array([[0,1,0],[1,-4,1],[0,1,0]], float),
        "Roberts X": np.array([[1,0,0],[0,-1,0],[0,0,0]], float),
        "Roberts Y": np.array([[0,1,0],[-1,0,0],[0,0,0]], float)
    }
    return kernels.get(op, np.zeros((3,3), float))

kernel = get_kernel(operation)

if operation == "Custom Kernel":
    st.markdown("### ✏️ Custom 3 × 3 Kernel Matrix")
    c1, c2, c3 = st.columns(3)
    with c1:
        a11 = st.number_input("K11", value=0.0)
        a21 = st.number_input("K21", value=0.0)
        a31 = st.number_input("K31", value=0.0)
    with c2:
        a12 = st.number_input("K12", value=0.0)
        a22 = st.number_input("K22", value=1.0)
        a32 = st.number_input("K32", value=0.0)
    with c3:
        a13 = st.number_input("K13", value=0.0)
        a23 = st.number_input("K23", value=0.0)
        a33 = st.number_input("K33", value=0.0)
    kernel = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]], dtype=float)


# ============================================================
# DEFAULT IMAGE GENERATOR
# ============================================================

def create_default_image():
    img = Image.new("L", (160,120), 235)
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,160,75], fill=210)
    d.ellipse([110,12,145,47], fill=250)
    d.polygon([(0,75),(35,35),(65,75)], fill=90)
    d.polygon([(40,75),(85,25),(135,75)], fill=120)
    d.rectangle([0,75,160,120], fill=150)
    d.rectangle([55,50,105,95], fill=75)
    return np.array(img)


# ============================================================
# KERNEL DISPLAY & UPLOAD INTERFACE
# ============================================================

kernel_col, upload_col = st.columns([0.8, 1.2])

with kernel_col:
    st.markdown('<div class="card"><div class="card-title">🔲 Active Kernel Matrix</div></div>', unsafe_allow_html=True)
    figk = go.Figure(data=go.Heatmap(
        z=kernel, text=np.round(kernel,3), texttemplate="%{text}",
        colorscale=[[0, "#ccfbf1"], [1, "#0d9488"]], showscale=False,
        x=["1","2","3"], y=["1","2","3"]
    ))
    figk.update_layout(height=200, margin=dict(l=15, r=15, t=5, b=15), font=dict(size=14))
    st.plotly_chart(figk, use_container_width=True)

with upload_col:
    st.markdown('<div class="card"><div class="card-title">🖼️ Input Image Source</div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload custom image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

if uploaded_file is not None:
    im = Image.open(uploaded_file).convert("L").resize((160,120))
    image_array = np.array(im)
    source_text = "Custom Upload"
else:
    image_array = create_default_image()
    source_text = "Synthetic Default Image"

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(f'<div class="info-box"><div class="info-title">Width</div><div class="info-value">{image_array.shape[1]} px</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown(f'<div class="info-box"><div class="info-title">Height</div><div class="info-value">{image_array.shape[0]} px</div></div>', unsafe_allow_html=True)
with i3:
    st.markdown(f'<div class="info-box"><div class="info-title">Format</div><div class="info-value">8-bit Grayscale</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="small-note">Source Active: <b>{source_text}</b> • Hover on interactive matrix below to sample pixel neighborhoods.</div>', unsafe_allow_html=True)


# ============================================================
# IMAGE PROCESSING ENGINE
# ============================================================

def process_image(img, op, k):
    if op == "Threshold":
        return np.where(img > 127, 255, 0).astype(np.uint8)
    elif op == "Invert":
        return (255 - img).astype(np.uint8)
    elif op == "Sobel Magnitude":
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
        ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float)
        gx = convolve2d(img, kx, mode="same", boundary="symm")
        gy = convolve2d(img, ky, mode="same", boundary="symm")
        res = np.sqrt(gx**2 + gy**2)
    else:
        res = convolve2d(img, k, mode="same", boundary="symm")
    
    res = np.abs(res)
    if res.max() > res.min():
        res = ((res - res.min()) / (res.max() - res.min())) * 255
    return np.clip(res, 0, 255).astype(np.uint8)

output_image = process_image(image_array, operation, kernel)


# ============================================================
# INPUT & OUTPUT DISPLAY
# ============================================================

st.markdown('<div class="section-title">🖼️ Workspace View</div>', unsafe_allow_html=True)
img_col1, img_col2 = st.columns(2)

with img_col1:
    st.markdown('<div class="card"><div class="card-title">Original Input</div></div>', unsafe_allow_html=True)
    st.image(image_array, use_container_width=True, clamp=True)

with img_col2:
    st.markdown(f'<div class="card"><div class="card-title">Processed Output ({operation})</div></div>', unsafe_allow_html=True)
    st.image(output_image, use_container_width=True, clamp=True)
    
    # New Feature: Download Processed Image Button
    out_pil = Image.fromarray(output_image)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Result Image",
        data=buf.getvalue(),
        file_name=f"processed_{operation.lower().replace(' ', '_')}.png",
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# INTERACTIVE KERNEL INSPECTION & LIVE CALCULATION
# ============================================================

if "cursor_row" not in st.session_state: st.session_state.cursor_row = image_array.shape[0] // 2
if "cursor_col" not in st.session_state: st.session_state.cursor_col = image_array.shape[1] // 2

r = max(1, min(st.session_state.cursor_row, image_array.shape[0] - 2))
c = max(1, min(st.session_state.cursor_col, image_array.shape[1] - 2))

interactive_col, calculation_col = st.columns([1, 1.25])

with interactive_col:
    st.markdown('<div class="section-title">🎯 Interactive Matrix Sampler</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=image_array, x=np.arange(image_array.shape[1]), y=np.arange(image_array.shape[0]),
        colorscale="Gray", zmin=0, zmax=255, showscale=False,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z}<extra></extra>"
    ))
    
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            fig.add_shape(type="rect", x0=c+dx-.5, x1=c+dx+.5, y0=r+dy-.5, y1=r+dy+.5, line=dict(color="#14b8a6", width=2), fillcolor="rgba(20,184,166,0.08)")

    fig.add_shape(type="rect", x0=c-.5, x1=c+.5, y0=r-.5, y1=r+.5, line=dict(color="#0f766e", width=4), fillcolor="rgba(15,118,110,0.15)")
    fig.update_layout(height=420, margin=dict(l=30, r=10, t=10, b=30), xaxis=dict(title="Column (X)", autorange="reversed"), yaxis=dict(title="Row (Y)", scaleanchor="x", scaleratio=1, autorange="reversed"))

    events = plotly_events(fig, hover_event=True, click_event=False, select_event=False, override_height=420, override_width="100%", key="live_image")
    if events:
        event = events[-1]
        if "x" in event and "y" in event:
            st.session_state.cursor_col = max(1, min(int(round(event["x"])), image_array.shape[1]-2))
            st.session_state.cursor_row = max(1, min(int(round(event["y"])), image_array.shape[0]-2))
            st.rerun()

with calculation_col:
    st.markdown('<div class="section-title">🧮 Real-time Neighborhood Convolution</div>', unsafe_allow_html=True)
    pixel_block = image_array[r-1:r+2, c-1:c+2].astype(float)
    multiplication = pixel_block * kernel
    convolution_sum = np.sum(multiplication)
    final_pixel = np.clip(abs(convolution_sum), 0, 255)

    calc1, calc2, calc3 = st.columns([1, 1, 0.8])

    with calc1:
        st.markdown("### ① Neighborhood")
        pf = go.Figure(data=go.Heatmap(z=pixel_block, text=pixel_block.astype(int), texttemplate="%{text}", colorscale="Greys", showscale=False))
        pf.update_layout(height=210, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(pf, use_container_width=True)

    with calc2:
        st.markdown("### ② Element Product")
        mf = go.Figure(data=go.Heatmap(z=multiplication, text=np.round(multiplication, 1), texttemplate="%{text}", colorscale=[[0, "#e0f2fe"], [1, "#0d9488"]], showscale=False))
        mf.update_layout(height=210, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(mf, use_container_width=True)

    with calc3:
        st.markdown("### ③ Output")
        st.metric(label="Sum Value", value=f"{convolution_sum:.1f}")
        st.metric(label="Pixel Out", value=f"{int(final_pixel)}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <span>VisionLab Interactive Engine</span> &nbsp;•&nbsp; 
    <span>Spatial Convolution Suite</span> &nbsp;•&nbsp; 
    <span>Edge Intelligence Laboratory</span>
</div>
""", unsafe_allow_html=True)    <span>Edge Intelligence Laboratory</span>
</div>
""", unsafe_allow_html=True)
