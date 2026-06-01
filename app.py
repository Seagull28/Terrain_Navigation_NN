# app.py  —  Gradio front-end for the TRN pipeline
# Drop this file in the ROOT of your repo alongside src/ and best.pt

import os
import io
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed on HF Spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from PIL import Image, ImageDraw

import gradio as gr

# ── Import your existing modules ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.NeuralNetwork import NeuralDetector
from src.LandingSystem  import LandingSystem

# Load model once at startup (not on every request)
detector = NeuralDetector(model_path="best.pt", conf_threshold=0.7)


# ── Helper: fig → PIL Image ───────────────────────────────────────────────────
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(pil_image):
    """
    Takes a PIL image, runs the full TRN pipeline, returns
    four output images and a metrics string.
    """

    if pil_image is None:
        return None, None, None, None, "⚠️ Please upload an image."

    # ── 1. Crater detection ──────────────────────────────────────────────────
    craters = detector.detectCratersNN(pil_image)

    if len(craters) < 4:
        return None, None, None, None, "⚠️ Not enough craters detected (need ≥ 4). Try a different image."

    img_arr = np.array(pil_image)
    landing = LandingSystem(img_arr.shape)

    # ── 2. Landing analysis ──────────────────────────────────────────────────
    best_point, score = landing.find_best_landing_point(craters)
    distance          = landing.distance_to_nearest_crater(best_point, craters)

    # ── 3. Localization image ────────────────────────────────────────────────
    loc_img  = pil_image.copy().convert("RGB")
    draw     = ImageDraw.Draw(loc_img)

    for crater in sorted(craters.values(), key=lambda c: c.score, reverse=True)[:10]:
        y, x = crater.centerpoint.astype(int)
        r    = int(crater.diameter / 2)
        draw.ellipse((x-r, y-r, x+r, y+r), outline="yellow", width=3)
        draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
        draw.text((x+5, y+5), f"d={crater.depth:.1f}", fill="yellow")

    mx, my = int(best_point[1]), int(best_point[0])
    draw.line((mx-12, my-12, mx+12, my+12), fill="lime", width=4)
    draw.line((mx-12, my+12, mx+12, my-12), fill="lime", width=4)

    # ── 4. Heatmap ───────────────────────────────────────────────────────────
    heatmap  = landing.generate_heatmap(craters)
    fig, ax  = plt.subplots(figsize=(7, 7))
    im_plot  = ax.imshow(heatmap, cmap="RdYlGn", origin="upper")
    fig.colorbar(im_plot, ax=ax, label="Landing Score (higher = safer)")
    ax.set_title("Landing Suitability Heatmap", fontsize=14, fontweight="bold")

    x_best, y_best = int(best_point[1]), int(best_point[0])
    ax.scatter(x_best, y_best, marker="x", s=250, c="black", linewidths=3, zorder=5)
    ax.add_patch(Circle((x_best, y_best), 40,
                         edgecolor="lime", facecolor="none", linewidth=2.5, zorder=4))

    offset_x = 70 if x_best < 300 else -130
    offset_y = -70 if y_best > 100 else 70
    ax.annotate(
        f"Best Landing Point\n({x_best}, {y_best})",
        xy=(x_best, y_best),
        xytext=(x_best + offset_x, y_best + offset_y),
        fontsize=8, color="black", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85),
    )
    for crater in craters.values():
        cy, cx = crater.centerpoint.astype(int)
        ax.add_patch(Circle((cx, cy), int(crater.diameter / 2),
                             edgecolor="white", facecolor="none",
                             linestyle="--", linewidth=1.5))
    heatmap_pil = fig_to_pil(fig)

    # ── 5. 3D Terrain ────────────────────────────────────────────────────────
    h = min(img_arr.shape[0], 512)
    w = min(img_arr.shape[1], 512)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z    = np.zeros((h, w))

    for crater in craters.values():
        cy, cx = crater.centerpoint
        cx     = np.clip(cx, 0, w - 1)
        cy     = np.clip(cy, 0, h - 1)
        radius = crater.diameter / 2.0
        Z     -= crater.depth * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * radius**2))

    by_img = int(np.clip(float(best_point[0]), 0, h - 1))
    bx_img = int(np.clip(float(best_point[1]), 0, w - 1))
    bz     = Z[by_img, bx_img]

    # Pin geometry
    z_head  = max(abs(Z.min()) * 0.20, 3.0)
    head_r  = max(z_head * 0.55, 8.0)
    theta_f = np.linspace(0, 2 * np.pi, 100)

    fig3 = plt.figure(figsize=(12, 9))
    ax3  = fig3.add_subplot(111, projection="3d")
    surf = ax3.plot_surface(X, Y, Z, cmap="terrain", linewidth=0,
                            antialiased=True, alpha=0.92)

    # Pole
    ax3.plot([bx_img, bx_img], [by_img, by_img], [bz, z_head],
             color="crimson", linewidth=4, label="Landing Point")
    # Disc head
    for frac in np.linspace(0.15, 1.0, 12):
        r = head_r * frac
        ax3.plot(bx_img + r * np.cos(theta_f),
                 by_img + r * np.sin(theta_f),
                 np.full(100, z_head),
                 color="crimson", linewidth=1.2, alpha=0.9)
    ax3.plot(bx_img + head_r * np.cos(theta_f),
             by_img + head_r * np.sin(theta_f),
             np.full(100, z_head), color="#8B0000", linewidth=2.0)
    ax3.scatter(bx_img, by_img, z_head, color="white", s=55,
                edgecolors="#8B0000", linewidths=1.5)
    ax3.text(bx_img, by_img, z_head + head_r + 0.6,
             f"  Landing\n({int(best_point[1])}, {int(best_point[0])})",
             color="crimson", fontsize=8, fontweight="bold",
             ha="center", va="bottom")
    # Shadow ring
    ax3.plot(bx_img + 15 * np.cos(theta_f),
             by_img + 15 * np.sin(theta_f),
             np.full(100, bz + 0.05),
             color="crimson", linewidth=1.5, linestyle="--",
             alpha=0.7, label="Safety Radius")

    ax3.view_init(elev=35, azim=-120)
    ax3.set_box_aspect([1, 1, 0.25])
    ax3.set_title("3D Terrain Reconstruction", fontsize=15, fontweight="bold")
    ax3.set_xlabel("X Coordinate", labelpad=8)
    ax3.set_ylabel("Y Coordinate", labelpad=8)
    ax3.set_zlabel("Elevation", labelpad=6)
    ax3.tick_params(axis="z", pad=4)
    ax3.legend(loc="upper right", fontsize=8)
    fig3.colorbar(surf, shrink=0.45, aspect=10, label="Terrain Elevation")
    terrain_pil = fig_to_pil(fig3)

    # ── 6. Density map ───────────────────────────────────────────────────────
    density_map = np.zeros((landing.height, landing.width), dtype=float)
    for crater in craters.values():
        cy, cx = crater.centerpoint.astype(int)
        r      = int(crater.diameter)
        density_map[
            max(0, cy - r): min(landing.height, cy + r),
            max(0, cx - r): min(landing.width,  cx + r)
        ] += 1

    fig4, ax4 = plt.subplots(figsize=(7, 7))
    im4 = ax4.imshow(density_map, cmap="hot", vmax=3)
    fig4.colorbar(im4, ax=ax4, label="Density")
    ax4.set_title("Crater Density Map", fontsize=14, fontweight="bold")
    density_pil = fig_to_pil(fig4)

    # ── 7. Metrics string ────────────────────────────────────────────────────
    crater_scores  = [c.score for c in craters.values()]
    metrics = (
        f"✅  Craters Detected : {len(craters)}\n"
        f"📍  Landing Point    : ({int(best_point[1])}, {int(best_point[0])})\n"
        f"🏆  Landing Score    : {round(float(score), 4)}\n"
        f"📏  Nearest Rim Dist : {round(distance, 1)} px\n"
        f"📊  Avg Confidence   : {round(float(np.mean(crater_scores)), 3)}\n"
        f"⬆️   Max Confidence   : {round(float(np.max(crater_scores)), 3)}\n"
        f"⬇️   Min Confidence   : {round(float(np.min(crater_scores)), 3)}"
    )

    return loc_img, heatmap_pil, terrain_pil, density_pil, metrics


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Terrain Relative Navigation — TRN System", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🛸 Autonomous Terrain Relative Navigation (TRN) System
        Upload a **lunar or planetary descent image** (.ppm / .jpg / .png).
        The pipeline detects craters using YOLOv8, scores every candidate
        landing zone, and selects the safest point automatically.
        """
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="📷 Upload Descent Image", height=350)

    run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

    with gr.Row():
        out_localization = gr.Image(label="🔍 Crater Detection & Landing Point")
        out_heatmap      = gr.Image(label="🌡️ Landing Suitability Heatmap")

    with gr.Row():
        out_terrain      = gr.Image(label="🏔️ 3D Terrain Reconstruction")
        out_density      = gr.Image(label="📊 Crater Density Map")

    out_metrics = gr.Textbox(label="📋 Mission Metrics", lines=8, interactive=False)

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_image],
        outputs=[out_localization, out_heatmap, out_terrain, out_density, out_metrics]
    )

    gr.Markdown(
        """
        ---
        **Model:** YOLOv8 fine-tuned on lunar crater imagery  
        **Scoring:** 5-term normalised function — clearance · centrality · density · smoothness · slope  
        **Source:** [github.com/Seagull28/Terrain_Navigation_NN](https://github.com/Seagull28/Terrain_Navigation_NN)
        """
    )

if __name__ == "__main__":
    demo.launch()