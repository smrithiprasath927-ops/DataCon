import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="AI Reconstruction Lab",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #0f1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.metric-box {
    background: #161b22;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
}

.big-font {
    font-size:18px !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TEXT → GRID
# =========================================================

def text_to_grid(text, size=8):
    text = text.ljust(size * size)[:size * size]
    ascii_vals = np.array([ord(c) for c in text], dtype=np.float32)
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
    noise = np.random.uniform(-sigma, sigma, grid.shape)
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

    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
    status_text = st.empty()

    for epoch in range(epochs):

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        progress = (epoch + 1) / epochs

        progress_bar.progress(progress)

        status_text.markdown(
            f"""
            <div class='metric-box'>
            <span class='big-font'>
            Epoch {epoch+1}/{epochs}
            </span><br>
            Current Loss: {loss.item():.4f}
            </div>
            """,
            unsafe_allow_html=True
        )

    return model, x, losses

# =========================================================
# PLOT GRID
# =========================================================

def plot_grid(grid, title):

    fig, ax = plt.subplots(figsize=(4,4))

    im = ax.imshow(
        grid,
        cmap="viridis"
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    return fig

# =========================================================
# ERROR HEATMAP
# =========================================================

def plot_error(original, reconstructed):

    error = np.abs(original - reconstructed)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.imshow(error, cmap="inferno")

    ax.set_title("Reconstruction Error")
    ax.axis("off")

    return fig

# =========================================================
# HEADER
# =========================================================

st.title("🧠 AI Data Reconstruction Laboratory")

st.markdown("""
This system explores whether AI can recover information after mathematical corruption using:
- ASCII matrix encoding
- Discrete diffusion
- Noise injection
- CNN-based reconstruction
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Simulation Controls")

text = st.sidebar.text_input(
    "Input Text",
    "SECRET MESSAGE"
)

steps = st.sidebar.slider(
    "Diffusion Steps",
    1,
    12,
    4
)

noise = st.sidebar.slider(
    "Noise Level",
    0,
    30,
    8
)

epochs = st.sidebar.slider(
    "Training Epochs",
    50,
    500,
    200
)

run = st.sidebar.button("🚀 Run Reconstruction")

# =========================================================
# MAIN EXECUTION
# =========================================================

if run:

    # ---------------------------------
    # ORIGINAL
    # ---------------------------------

    clean = text_to_grid(text)

    # ---------------------------------
    # DIFFUSION ANIMATION
    # ---------------------------------

    st.subheader("🔄 Diffusion Process")

    animation_placeholder = st.empty()

    final_diffused, history = diffuse(clean, steps)

    for idx, frame in enumerate(history):

        fig = plot_grid(
            frame,
            f"Diffusion Step {idx}"
        )

        animation_placeholder.pyplot(fig)

        time.sleep(0.35)

    # ---------------------------------
    # ADD NOISE
    # ---------------------------------

    corrupted = add_noise(final_diffused, noise)

    # ---------------------------------
    # TRAIN
    # ---------------------------------

    st.subheader("🧠 CNN Reconstruction")

    model, x, losses = train_model(
        clean,
        corrupted,
        epochs
    )

    # ---------------------------------
    # PREDICT
    # ---------------------------------

    with torch.no_grad():
        output = model(x).squeeze().numpy()

    # ---------------------------------
    # METRICS
    # ---------------------------------

    mse = compute_mse(clean, output)
    similarity = compute_similarity(clean, output)

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"""
        <div class='metric-box'>
        <h3>MSE</h3>
        <h2>{mse:.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class='metric-box'>
        <h3>Similarity</h3>
        <h2>{similarity:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class='metric-box'>
        <h3>Diffusion Steps</h3>
        <h2>{steps}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------
    # GRID VISUALS
    # ---------------------------------

    st.subheader("📊 Grid Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(plot_grid(clean, "Original Grid"))

    with col2:
        st.pyplot(plot_grid(corrupted, "Corrupted Grid"))

    with col3:
        st.pyplot(plot_grid(output, "Reconstructed Grid"))

    # ---------------------------------
    # ERROR MAP
    # ---------------------------------

    st.subheader("🔥 Reconstruction Error Heatmap")

    st.pyplot(
        plot_error(clean, output)
    )

    # ---------------------------------
    # LOSS CURVE
    # ---------------------------------

    st.subheader("📉 Training Loss")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(losses)

    ax.set_title("MSE Loss Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    st.pyplot(fig)

    # ---------------------------------
    # TEXT COMPARISON
    # ---------------------------------

    st.subheader("📝 Text Recovery")

    recovered_text = grid_to_text(output)

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("### Original")
        st.code(text)

    with t2:
        st.markdown("### Recovered")
        st.code(recovered_text)

    # ---------------------------------
    # CONCLUSION
    # ---------------------------------

    st.success(
        "Reconstruction complete. The CNN attempted to learn the inverse mapping from corrupted data back to the original structure."
    )