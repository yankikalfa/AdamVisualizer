# app.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf

# 🛑 Must be the first Streamlit call!
st.set_page_config(layout="wide")

st.title("🎯 Adam Optimizer Trajectory Explorer")
st.markdown("Explore how different **learning rates**, **loss surfaces**, and **user-placed attractors** affect optimizer convergence!")

# ---- UI for surface + optimizer ----
# 🚨 Handle edge case: no learning rates selected
if 'selected_lrs' in locals() and selected_lrs == []:
    st.warning("Please select at least one learning rate to visualize.")
    st.stop()
surface_choice = st.selectbox("Choose Loss Surface", ["Twin Basins", "Multi Gaussians", "Wavy + Dips", "Funnel Pit", "Custom Attractors"])
noise_level = st.slider("Surface Noise / Difficulty", 0.0, 1.0, 0.0, step=0.1)

col1, col2 = st.columns(2)
with col1:
    preset_lrs = [0.0001, 0.001, 0.01, 0.1]
    selected_lrs = st.multiselect("Compare Learning Rates", preset_lrs, default=[0.001, 0.01])
    steps = st.slider("Steps", 50, 500, 200, step=25)
with col2:
    x0 = st.slider("Starting X", -1.0, 1.0, value=0.0, step=0.05)
    y0 = st.slider("Starting Y", -1.0, 1.0, value=-1.0, step=0.05)

# ---- Attractors for Custom Surface ----
# 🚨 Handle empty custom attractors if surface is selected
if surface_choice == "Custom Attractors" and not custom_attractors:
    st.warning("Define at least one attractor to visualize the custom surface.")
    st.stop()
custom_attractors = []
if surface_choice == "Custom Attractors":
    st.markdown("### 🎯 Define Custom Attractors")
    num = st.slider("Number of Attractors", 1, 5, 2)
    for i in range(num):
        col1, col2, col3 = st.columns(3)
        with col1:
            x_a = st.slider(f"X{i+1}", -1.2, 1.2, -0.4 + 0.8 * i / max(num-1, 1), key=f"x{i}")
        with col2:
            y_a = st.slider(f"Y{i+1}", -1.2, 1.2, 0.0, key=f"y{i}")
        with col3:
            depth = st.slider(f"Depth{i+1}", 0.5, 2.0, 1.0, key=f"d{i}")
        custom_attractors.append((x_a, y_a, depth))

# ---- Cost Surface Function ----
def f2(x, y, x0, y0, sx, sy):
    return np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2))) / (2 * np.pi * sx * sy)

def cost_function_np(x, y):
    z = 0
    if surface_choice == "Twin Basins":
        z -= 1.0 * f2(x, y, -0.4, 0.0, 0.2, 0.2)
        z -= 1.3 * f2(x, y, 0.4, 0.0, 0.2, 0.2)
        z -= 0.4 * f2(x, y, 0.0, 0.0, 0.25, 0.25)
    elif surface_choice == "Multi Gaussians":
        z -= 1.2 * f2(x, y, -0.8, 0.6, 0.2, 0.2)
        z -= 0.9 * f2(x, y, 0.8, 0.6, 0.3, 0.3)
        z -= 1.5 * f2(x, y, 0.0, -0.8, 0.2, 0.2)
        z -= 0.3 * f2(x, y, 0.0, 0.0, 0.3, 0.3)
    elif surface_choice == "Wavy + Dips":
        z = -np.sin(x**2 + y**2) * np.exp(-0.1 * (x**2 + y**2))
        z -= 1.0 * f2(x, y, 0.5, 0.5, 0.2, 0.2)
        z -= 1.0 * f2(x, y, -0.5, -0.5, 0.2, 0.2)
    elif surface_choice == "Funnel Pit":
        z = -np.exp(-(x**2 + y**2) / 0.05)
    elif surface_choice == "Custom Attractors":
        for x_a, y_a, d in custom_attractors:
            z -= d * f2(x, y, x_a, y_a, 0.2, 0.2)

    if noise_level > 0:
        z += noise_level * np.sin(5 * x) * np.cos(5 * y)
    return z

def f2_tf(x, y, x0, y0, sx, sy):
    norm = 1 / (2 * np.pi * sx * sy)
    x_exp = -((x - x0) ** 2) / (2 * sx ** 2)
    y_exp = -((y - y0) ** 2) / (2 * sy ** 2)
    return norm * tf.exp(x_exp + y_exp)

def cost_function_tf(x, y):
    z = 0.0
    if surface_choice == "Twin Basins":
        z -= 1.0 * f2_tf(x, y, -0.4, 0.0, 0.2, 0.2)
        z -= 1.3 * f2_tf(x, y, 0.4, 0.0, 0.2, 0.2)
        z -= 0.4 * f2_tf(x, y, 0.0, 0.0, 0.25, 0.25)
    elif surface_choice == "Multi Gaussians":
        z -= 1.2 * f2_tf(x, y, -0.8, 0.6, 0.2, 0.2)
        z -= 0.9 * f2_tf(x, y, 0.8, 0.6, 0.3, 0.3)
        z -= 1.5 * f2_tf(x, y, 0.0, -0.8, 0.2, 0.2)
        z -= 0.3 * f2_tf(x, y, 0.0, 0.0, 0.3, 0.3)
    elif surface_choice == "Wavy + Dips":
        r2 = x ** 2 + y ** 2
        z = -tf.sin(r2) * tf.exp(-0.1 * r2)
        z -= 1.0 * f2_tf(x, y, 0.5, 0.5, 0.2, 0.2)
        z -= 1.0 * f2_tf(x, y, -0.5, -0.5, 0.2, 0.2)
    elif surface_choice == "Funnel Pit":
        z = -tf.exp(-(x ** 2 + y ** 2) / 0.05)
    elif surface_choice == "Custom Attractors":
        for x_a, y_a, d in custom_attractors:
            z -= d * f2_tf(x, y, x_a, y_a, 0.2, 0.2)

    if noise_level > 0:
        z += noise_level * tf.sin(5 * x) * tf.cos(5 * y)
    return z

# ---- Plot 3D Surface ----
x_vals = y_vals = np.linspace(-1.5, 1.5, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = cost_function_np(X, Y)

fig = go.Figure()
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='RdBu', opacity=0.6, showscale=False))
fig.add_trace(go.Contour(z=Z, x=x_vals, y=y_vals, contours_coloring='lines', showscale=False, line=dict(width=1), opacity=0.5))

colors = ['black', 'blue', 'green', 'red', 'orange']
for idx, lr in enumerate(selected_lrs):
    x = tf.Variable(x0, dtype=tf.float32)
    y = tf.Variable(y0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    history = [(x0, y0, cost_function_np(x0, y0))]
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = cost_function_tf(x, y)
        grads = tape.gradient(loss, [x, y])
        grads_and_vars = [(g, v) for g, v in zip(grads, [x, y]) if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)
        else:
            # Automatically reset to random nearby position
            new_x = x.numpy() + np.random.uniform(-0.1, 0.1)
            new_y = y.numpy() + np.random.uniform(-0.1, 0.1)
            x.assign(new_x)
            y.assign(new_y)
            st.info(f"🔁 Resetting optimizer to new position (x={new_x:.2f}, y={new_y:.2f}) due to vanishing gradients.")
        history.append((x.numpy(), y.numpy(), cost_function_np(x.numpy(), y.numpy())))

    xs, ys, zs = zip(*history)
    color = colors[idx % len(colors)]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines+markers', name=f"lr={lr}",
                                line=dict(color=color, width=4),
                                marker=dict(size=4, color=color)))
    fig.add_trace(go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode='markers',
                                marker=dict(size=8, color=color), name=f"Final lr={lr}"))

fig.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10),
                  scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Loss",
                             camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))))

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**Surface:** `{surface_choice}`  \
**Starting Point:** ({x0:.2f}, {y0:.2f})  \
**Learning Rates:** {', '.join(map(str, selected_lrs))}  \
**Steps:** {steps}")
