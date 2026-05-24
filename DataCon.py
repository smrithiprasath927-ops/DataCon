import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DataCon",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    color: white;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(145deg, #111827, #1f2937);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.25);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    color: #9ca3af;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class='main-title'>
🧠 DataCon
</div>
""", unsafe_allow_html=True)

st.markdown("""
This system studies whether AI can recover structured information after mathematical corruption using:
- Discrete diffusion
- Noise injection
- Convolutional Neural Networks
""")

st.markdown("---")

# =========================================================
# TEXT → GRID
# =========================================================

def text_to_grid(text, size=8):

    text = text.ljust(size * size)[:size * size]

    ascii_vals = np.array(
        [ord(c) for c in text],
        dtype=np.float32
    )

    return ascii_vals.reshape(size, size)

def grid_to_text(grid):

    flat = np.clip(grid.flatten(), 32, 126)

    return ''.join(chr(int(x)) for x in flat)

# =========================================================
# DIFFUSION
# =========================================================

def diffuse_step(grid):

    g = grid.copy()
    new_g = g.copy()

    for i in range(1, g.shape[0]-1):
        for j in range(1, g.shape[1]-1):

            neighbors = [
                g[i,j],
                g[i-1,j],
                g[i+1,j],
                g[i,j-1],
                g[i,j+1]
            ]

            new_g[i,j] = sum(neighbors)/len(neighbors)

    return new_g

def diffuse(grid, steps):

    history = [grid.copy()]

    g = grid.copy()

    for _ in range(steps):

        g = diffuse_step(g)

        history.append(g.copy())

    return g, history

# =========================================================
# NOISE
# =========================================================

def add_noise(grid, sigma):

    noise = np.random.uniform(
        -sigma,
        sigma,
        grid.shape
    )

    return grid + noise

# =========================================================
# METRICS
# =========================================================

def compute_mse(a, b):

    return np.mean((a - b) ** 2)

def compute_similarity(a, b):

    diff = np.mean(np.abs(a - b))

    similarity = max(0, 100 - diff)

    return similarity

# =========================================================
# CNN MODEL
# =========================================================

class DenoiserCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):

        return self.net(x)

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(clean, corrupted, epochs):

    model = DenoiserCNN()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    loss_fn = nn.MSELoss()

    x = torch.tensor(
        corrupted,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    y = torch.tensor(
        clean,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    losses = []

    progress_bar = st.progress(0)

    live_chart = st.empty()

    for epoch in range(epochs):

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        progress_bar.progress((epoch+1)/epochs)

        # LIVE LOSS CHART
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=losses,
            mode='lines',
            line=dict(width=3)
        ))

        fig.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=20,r=20,t=20,b=20),
            title="Live Training Loss"
        )

        live_chart.plotly_chart(
            fig,
            use_container_width=True
        )

    return model, x, losses

# =========================================================
# PLOTLY HEATMAP
# =========================================================

def heatmap(grid, title):

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=350,
        margin=dict(l=10,r=10,t=40,b=10)
    )

    return fig

# =========================================================
# ERROR MAP
# =========================================================

def error_heatmap(a, b):

    error = np.abs(a - b)

    fig = go.Figure(data=go.Heatmap(
        z=error,
        colorscale='Inferno'
    ))

    fig.update_layout(
        title="Reconstruction Error",
        template="plotly_dark",
        height=350,
        margin=dict(l=10,r=10,t=40,b=10)
    )

    return fig

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙ Controls")

    text = st.text_input(
        "Input Text",
        "SECRET MESSAGE"
    )

    steps = st.slider(
        "Diffusion Steps",
        1,
        12,
        4
    )

    noise = st.slider(
        "Noise Level",
        0,
        30,
        8
    )

    epochs = st.slider(
        "Training Epochs",
        50,
        300,
        150
    )

    run = st.button(
        "🚀 Run Reconstruction",
        use_container_width=True
    )

# =========================================================
# MAIN EXECUTION
# =========================================================

if run:

    clean = text_to_grid(text)

    st.markdown("## 🔄 Diffusion Animation")

    final_diffused, history = diffuse(clean, steps)

    animation_placeholder = st.empty()

    for idx, frame in enumerate(history):

        fig = heatmap(
            frame,
            f"Diffusion Step {idx}"
        )

        animation_placeholder.plotly_chart(
            fig,
            use_container_width=True
        )

        time.sleep(0.3)

    corrupted = add_noise(
        final_diffused,
        noise
    )

    st.markdown("---")

    st.markdown("## 🧠 CNN Reconstruction Training")

    model, x, losses = train_model(
        clean,
        corrupted,
        epochs
    )

    with torch.no_grad():

        output = model(x).squeeze().numpy()

    # =====================================================
    # METRICS
    # =====================================================

    mse = compute_mse(clean, output)

    similarity = compute_similarity(
        clean,
        output
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-value'>{mse:.2f}</div>
        <div class='metric-label'>MSE Loss</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-value'>{similarity:.1f}%</div>
        <div class='metric-label'>Similarity</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-value'>{steps}</div>
        <div class='metric-label'>Diffusion Steps</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # MAIN VISUALIZATION
    # =====================================================

    st.markdown("## 📊 Reconstruction Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            heatmap(clean, "Original Grid"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            heatmap(corrupted, "Corrupted Grid"),
            use_container_width=True
        )

    with col3:
        st.plotly_chart(
            heatmap(output, "Reconstructed Grid"),
            use_container_width=True
        )

    # =====================================================
    # ERROR MAP
    # =====================================================

    st.markdown("## 🔥 Error Heatmap")

    st.plotly_chart(
        error_heatmap(clean, output),
        use_container_width=True
    )

    # =====================================================
    # LOSS CURVE
    # =====================================================

    st.markdown("## 📉 Training Curve")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=losses,
        mode='lines',
        line=dict(width=4)
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Model Loss Over Time",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # TEXT RECOVERY
    # =====================================================

    st.markdown("## 📝 Text Reconstruction")

    recovered_text = grid_to_text(output)

    t1, t2 = st.columns(2)

    with t1:

        st.markdown("""
        <div class='glass'>
        <h3>Original</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code(text)

    with t2:

        st.markdown("""
        <div class='glass'>
        <h3>Recovered</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code(recovered_text)

    st.markdown("---")

    st.success(
        "Reconstruction complete. The CNN learned an approximate inverse mapping from corrupted structured data back to the original representation."
    )