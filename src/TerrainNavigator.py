# src/TerrainNavigator.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
from datetime import datetime

from src.NeuralNetwork import NeuralDetector
from src.LandingSystem import LandingSystem
from src.ExperimentLogger import ExperimentLogger


class Navigator:

    def __init__(self, referenceAltitude, referenceMap,
                 referenceCatalogue, datapath):

        self.datapath = datapath

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("outputs", timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = ExperimentLogger(self.output_dir)
        self.detector = NeuralDetector() 
        print(f"📁 Saving outputs to: {self.output_dir}")

    # ----------------------------------------
    # DRAW LOCALIZATION MAP
    # ----------------------------------------
    def drawDescentImageOnReferenceImage(self, best_point, descentImageCraters):
        """
        Overlay detected craters and the best landing point
        on the raw descent image. Returns the finalized PIL Image object.
        """
        im   = Image.open(self.currentDescentImage).convert("RGB")
        draw = ImageDraw.Draw(im)

        craters_sorted = sorted(
            descentImageCraters.values(),
            key=lambda c: getattr(c, "score", 0),
            reverse=True
        )[:10]

        for crater in craters_sorted:
            y, x = crater.centerpoint.astype(int)
            r    = int(crater.diameter / 2)

            # Crater boundary
            draw.ellipse((x-r, y-r, x+r, y+r), outline='yellow', width=3)

            # Crater centre dot
            draw.ellipse((x-2, y-2, x+2, y+2), fill='red')

            # Label: depth (1 dp)
            draw.text(
                (x + 5, y + 5),
                f"d={crater.depth:.1f}",
                fill="yellow"
            )

        # Best landing point — green X
        mx = int(best_point[1])
        my = int(best_point[0])

        draw.line((mx-10, my-10, mx+10, my+10), fill='lime', width=4)
        draw.line((mx-10, my+10, mx+10, my-10), fill='lime', width=4)

        path = os.path.join(self.output_dir, "localization.png")
        im.save(path)
        print(f"🖼️  Saved: {path}")
        return im

    # ----------------------------------------
    # DRAW DISTANCE MAP
    # ----------------------------------------
    def drawCraterDistances(self, descentImageCraters):
        """Connect the top-5 craters by confidence with distance labels. Returns PIL Image."""
        im   = Image.open(self.currentDescentImage).convert("RGB")
        draw = ImageDraw.Draw(im)

        craters = sorted(
            descentImageCraters.values(),
            key=lambda c: getattr(c, "score", 0),
            reverse=True
        )[:5]

        for i in range(len(craters)):
            for j in range(i + 1, len(craters)):

                c1 = craters[i]
                c2 = craters[j]

                y1, x1 = c1.centerpoint.astype(int)
                y2, x2 = c2.centerpoint.astype(int)

                dist = np.linalg.norm(c1.centerpoint - c2.centerpoint)

                draw.line((x1, y1, x2, y2), fill="cyan", width=2)

                mx = int((x1 + x2) / 2)
                my = int((y1 + y2) / 2)

                draw.text((mx, my), f"{int(dist)}", fill="white")

        path = os.path.join(self.output_dir, "distances.png")
        im.save(path)
        print(f"🖼️  Saved: {path}")
        return im

    # ----------------------------------------
    # MAIN PIPELINE (UPDATED FOR STREAMLIT FLOW)
    # ----------------------------------------
    def locateDescentImageInReferenceImage(self, imagename):
        """
        Runs the full TRN pipeline.
        Returns: best_point, fig_heatmap, fig_density, fig_3d, im_localization
        """
        self.currentDescentImage = imagename
        im = Image.open(imagename)

        descentImageCraters = self.detector.detectCratersNN(im)

        if len(descentImageCraters) < 2:
            print("⚠️  Not enough craters detected (need ≥ 2)")
            return None

        # ---- Detection statistics ----
        small_craters  = sum(1 for c in descentImageCraters.values() if c.diameter < 20)
        low_confidence = sum(1 for c in descentImageCraters.values() if c.score < 0.8)

        print("\n⚠️  DETECTION ANALYSIS")
        print("----------------------")
        print(f"Small Craters        : {small_craters}")
        print(f"Low Confidence (<0.8): {low_confidence}")

        # ---- Landing analysis (Returns figures directly) ----
        best_point, landing_score, fig_heatmap, fig_density = self.analyzeLandingSafety(
            im, descentImageCraters
        )

        # ---- Report Logging ----
        crater_scores  = [c.score for c in descentImageCraters.values()]
        avg_confidence = np.mean(crater_scores)
        max_confidence = np.max(crater_scores)
        min_confidence = np.min(crater_scores)

        best_point_py = (int(best_point[0]), int(best_point[1]))
        self.logger.log(f"Landing Point: {best_point_py}")
        self.logger.log(f"Landing Score: {landing_score}")
        self.logger.log("\nPERFORMANCE METRICS")
        self.logger.log("----------------------")
        self.logger.log(f"Total Craters Detected: {len(descentImageCraters)}")
        self.logger.log(f"Average Confidence: {round(avg_confidence, 3)}")
        self.logger.log(f"Maximum Confidence: {round(max_confidence, 3)}")
        self.logger.log(f"Minimum Confidence: {round(min_confidence, 3)}")

        # ---- Visualisations Generation ----
        fig_3d = self.generate3DTerrainMap(descentImageCraters, best_point)
        im_localization = self.drawDescentImageOnReferenceImage(best_point, descentImageCraters)
        self.drawCraterDistances(descentImageCraters)

        print(f"\n✅ Localization complete — best landing point: {tuple(best_point)}")
        
        # Streamlit unpack target return sequence
        return best_point, fig_heatmap, fig_density, fig_3d, im_localization

    # ----------------------------------------
    # LANDING HEATMAP + DENSITY MAP
    # ----------------------------------------
    def analyzeLandingSafety(self, im, descentImageCraters):

        img_array = np.array(im)
        landing   = LandingSystem(img_array.shape)

        best_point, score = landing.find_best_landing_point(descentImageCraters)
        distance = landing.distance_to_nearest_crater(best_point, descentImageCraters)

        print("\n🎯 LANDING ANALYSIS")
        print("----------------------")
        print(f"Best Landing Point         : {tuple(best_point)}")
        print(f"Landing Score              : {round(float(score), 4)}")
        print(f"Distance from nearest rim  : {round(distance, 2)} px")

        heatmap = landing.generate_heatmap(descentImageCraters)

        # ---- Heatmap figure ----
        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(heatmap, cmap='RdYlGn', origin='upper')
        fig.colorbar(img, ax=ax, label="Landing Score (higher = safer)")
        ax.set_title("Landing Suitability Heatmap", fontsize=16, fontweight='bold')

        x_best = int(best_point[1])
        y_best = int(best_point[0])

        ax.scatter(x_best, y_best, marker='x', s=300, c='black', linewidths=3, zorder=5)

        x_offset = -60 if x_best > (landing.width / 2) else 60
        y_offset = -40 if y_best > (landing.height / 2) else 40

        ax.annotate(
            f"Best Landing Point\n({x_best}, {y_best})",
            xy=(x_best, y_best),
            xytext=(x_best + x_offset, y_best + y_offset),
            arrowprops=dict(facecolor='black', shrink=0.08, width=2, headwidth=8),
            fontsize=10,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9),
            ha='center', va='center',
            zorder=6
        )

        ax.add_patch(Circle((x_best, y_best), 40, edgecolor='lime', facecolor='none', linewidth=3, zorder=4))

        for crater in descentImageCraters.values():
            y, x = crater.centerpoint.astype(int)
            r    = int(crater.diameter / 2)
            ax.add_patch(Circle((x, y), r, edgecolor='white', facecolor='none', linestyle='--', linewidth=2))

        heatmap_path = os.path.join(self.output_dir, "heatmap.png")
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"🖼️  Saved: {heatmap_path}")

        # ---- Upgraded Circular Density map ----
        density_map = np.zeros((landing.height, landing.width), dtype=float)

        for crater in descentImageCraters.values():
            cy, cx = crater.centerpoint
            r = crater.diameter / 2.0

            y_min = max(0, int(np.floor(cy - r)))
            y_max = min(landing.height, int(np.ceil(cy + r)))
            x_min = max(0, int(np.floor(cx - r)))
            x_max = min(landing.width, int(np.ceil(cx + r)))

            y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
            distance_squared = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
            circular_mask = distance_squared <= (r ** 2)
            density_map[y_min:y_max, x_min:x_max] += circular_mask

        fig2, ax2 = plt.subplots(figsize=(8, 8))
        img2 = ax2.imshow(density_map, cmap='hot', origin='upper')
        fig2.colorbar(img2, ax=ax2, label="Density")
        ax2.set_title("Crater Density Map", fontsize=16, fontweight='bold')

        density_path = os.path.join(self.output_dir, "density_map.png")
        fig2.savefig(density_path, dpi=300, bbox_inches='tight')
        print(f"🖼️  Saved upgraded circular density map: {density_path}")

        return best_point, score, fig, fig2

    # ----------------------------------------
    # 3D TERRAIN MAP
    # ----------------------------------------
    def generate3DTerrainMap(self, descentImageCraters, best_point):
        """Reconstructs a 3-D depth surface and returns the active figure object."""
        im_arr = np.array(Image.open(self.currentDescentImage))
        h = min(im_arr.shape[0], 512)
        w = min(im_arr.shape[1], 512)

        x = np.arange(0, w)
        y = np.arange(0, h)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros((h, w))
        for crater in descentImageCraters.values():
            cy, cx = crater.centerpoint
            radius = crater.diameter / 2.0
            depth  = crater.depth

            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)

            Z -= depth * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * radius ** 2))

        by_img = int(np.clip(float(best_point[0]), 0, h - 1))
        bx_img = int(np.clip(float(best_point[1]), 0, w - 1))
        bz     = Z[by_img, bx_img]

        fig = plt.figure(figsize=(14, 11))
        ax  = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=True, alpha=0.92, zorder=1)

        z_surface   = bz
        z_head      = max(abs(Z.min()) * 0.20, 3.0)
        theta_full  = np.linspace(0, 2 * np.pi, 100)

        ax.plot([bx_img, bx_img], [by_img, by_img], [z_surface, z_head], color='crimson', linewidth=4, solid_capstyle='round', zorder=10, label='Landing Point')
        ax.scatter([bx_img], [by_img], [z_head], color='crimson', s=250, edgecolors='white', linewidths=2, zorder=11)
        ax.text(bx_img, by_img, z_head + 1.0, f'  Landing\n({int(best_point[1])}, {int(best_point[0])})', color='crimson', fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=12)

        shadow_r = 15
        ax.plot(bx_img + shadow_r * np.cos(theta_full), by_img + shadow_r * np.sin(theta_full), np.full(100, z_surface + 0.15), color='crimson', linewidth=2, linestyle='--', alpha=0.8, zorder=10, label='Landing Safety Radius')

        ax.view_init(elev=35, azim=-120)
        ax.set_box_aspect([1, 1, 0.25])
        ax.set_title("3D Terrain Reconstruction", fontsize=18, fontweight='bold')
        ax.set_xlabel("X Coordinate", labelpad=10)
        ax.set_ylabel("Y Coordinate", labelpad=10)
        ax.set_zlabel("Elevation", labelpad=8)
        ax.tick_params(axis='z', pad=5)
        ax.legend(loc='upper right', fontsize=9)

        fig.colorbar(surf, shrink=0.45, aspect=10, label='Terrain Elevation')

        path = os.path.join(self.output_dir, "terrain_3d.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')

        if os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND", "").lower() == "tkagg":
            plt.show()

        return fig