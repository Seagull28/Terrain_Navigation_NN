import numpy as np
from PIL import Image, ImageDraw
from src.NeuralNetwork import detectCratersNN
from src.LandingSystem import LandingSystem
import matplotlib.pyplot as plt
import os
from datetime import datetime


class Navigator:
    def __init__(self, referenceAltitude, referenceMap, referenceCatalogue, datapath):
        self.datapath = datapath

        # 🔥 Create output folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("outputs", timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 Saving outputs to: {self.output_dir}")

    # ----------------------------------------
    def drawDescentImageOnReferenceImage(self, middle, descentImageCraters):
        im = Image.open(self.currentDescentImage)
        draw = ImageDraw.Draw(im)

        # 🔥 Draw only TOP craters (clean depth)
        craters_sorted = sorted(
            descentImageCraters.values(),
            key=lambda c: getattr(c, "score", 0),
            reverse=True
        )[:10]

        for crater in craters_sorted:
            y, x = crater.centerpoint.astype(int)
            r = int(crater.diameter / 2)

            draw.ellipse((x - r, y - r, x + r, y + r), outline='yellow', width=2)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red')

            if hasattr(crater, "depth"):
                draw.text((x + 5, y + 5), f"d={round(crater.depth,1)}", fill="yellow")

        # Lander position
        mx = int(middle[1])
        my = int(middle[0])

        draw.ellipse(
            (
                mx - 5,
                my - 5,
                mx + 5,
                my + 5
            ),
            fill='green'
        )   

        # Save
        path = os.path.join(self.output_dir, "localization.png")
        im.save(path)
        print(f"🖼️ Saved: {path}")

        im.show()

    # ----------------------------------------
    def drawCraterDistances(self, descentImageCraters):
        im = Image.open(self.currentDescentImage)
        draw = ImageDraw.Draw(im)

        craters = sorted(
            descentImageCraters.values(),
            key=lambda c: getattr(c, "score", 0),
            reverse=True
        )[:5]  # 🔥 only top 5

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
        print(f"🖼️ Saved: {path}")

        im.show()

    # ----------------------------------------
    def locateDescentImageInReferenceImage(self, imagename):

        self.currentDescentImage = imagename

        im = Image.open(imagename)
        descentImageCraters = detectCratersNN(im)

        if len(descentImageCraters) < 4:
            print("⚠️ Not enough craters")
            return None

        # 🔥 Stable center calculation
        centers = np.array([c.centerpoint for c in descentImageCraters.values()])
        middle = np.median(centers, axis=0)

        best_point, landing_score = self.analyzeLandingSafety(
            im,
            descentImageCraters
        )   
        self.generate3DTerrainMap(
            descentImageCraters,
            best_point
        )
        self.drawDescentImageOnReferenceImage(
            best_point,
            descentImageCraters
        )   
        self.drawCraterDistances(descentImageCraters)

        print("✅ Localization successful:", middle)

    # ----------------------------------------
    # SAFE LANDING ANALYSIS
    # ----------------------------------------  

    def analyzeLandingSafety(self, im, descentImageCraters):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        landing = LandingSystem(np.array(im).shape)

        best_point, score = landing.find_best_landing_point(
            descentImageCraters
        )

        distance = landing.distance_to_nearest_crater(
            best_point,
            descentImageCraters
        )   

        print("\n🎯 LANDING ANALYSIS")
        print("----------------------")
        print("Best Landing Point:", best_point)
        print("Landing Score:", round(score, 2))
        print("Distance from nearest crater:", round(distance, 2), "px")

        # -------------------------------------------------
        # GENERATE HEATMAP
        # -------------------------------------------------
        heatmap = landing.generate_heatmap(
            descentImageCraters
        )   

        plt.figure(figsize=(8, 8))

        plt.imshow(
            heatmap,
            cmap='jet',
            origin='upper'
        )

        plt.colorbar(label="Landing Score") 

        plt.title(
            "Landing Suitability Heatmap",
            fontsize=16,
            fontweight='bold'
        )

        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")    

        # -------------------------------------------------
        # MARK BEST LANDING POINT
        # -------------------------------------------------
        x_best = int(best_point[1])
        y_best = int(best_point[0]) 

        plt.scatter(
            x_best,
            y_best,
            marker='x',
            s=300,
            c='black',
            linewidths=3,
            label='Best Landing Point'
        )

        # -------------------------------------------------
        # DRAW CRATER BOUNDARIES
        # -------------------------------------------------
        for crater in descentImageCraters.values():     

            y, x = crater.centerpoint.astype(int)

            r = int(crater.diameter / 2)

            circle = Circle(
                (x, y),
                r,
                edgecolor='white',
                facecolor='none',
                linestyle='--',
                linewidth=2
            )

            plt.gca().add_patch(circle)

        # -------------------------------------------------
        # LEGEND
        # -------------------------------------------------
        plt.legend(loc='upper left')    

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------
        heatmap_path = os.path.join(
           self.output_dir,
            "heatmap.png"
        )

        plt.savefig(
            heatmap_path,
            dpi=300
        )   

        print(f"🖼️ Saved: {heatmap_path}")

        plt.show()

        return best_point, score

    # ----------------------------------------
    # IMPROVED 3D TERRAIN VISUALIZATION
    # ----------------------------------------  
    def generate3DTerrainMap(self,descentImageCraters,best_point):  

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import os   

        # ----------------------------------------
        # TERRAIN SIZE
        # ----------------------------------------
        h = 512
        w = 512 

        # Elevation map
        Z = np.zeros((h, w))

        # Coordinate grid
        x = np.arange(0, w)
        y = np.arange(0, h)

        X, Y = np.meshgrid(x, y)

        # ----------------------------------------
        # CREATE CRATER DEPRESSIONS
        # ----------------------------------------
        for crater in descentImageCraters.values(): 

            cy, cx = crater.centerpoint

            radius = crater.diameter / 2

            depth = crater.depth

            # Gaussian depression   
            crater_surface = -depth * np.exp(
                -(
                    ((X - cx) ** 2 + (Y - cy) ** 2)
                    / (2 * (radius ** 2))
                )
            )

            Z += crater_surface

        # ----------------------------------------
        # PLOT
        # ----------------------------------------
        fig = plt.figure(figsize=(12, 10))

        ax = fig.add_subplot(111, projection='3d')

        surface = ax.plot_surface(
            X,
            Y,
            Z,
            cmap='terrain',
            linewidth=0,
            antialiased=True
        )   


        # ----------------------------------------
        # LANDING POINT
        # ----------------------------------------

        # Ensure clean coordinates
        by = int(float(best_point[0]))
        bx = int(float(best_point[1]))  

        # Clamp inside terrain bounds
        by = max(0, min(by, h - 1))
        bx = max(0, min(bx, w - 1))

        bz = Z[by, bx]  

        ax.scatter(
            bx,
            by,
            bz + 2,
            color='red',
            s=200,
            marker='X',
            label='Landing Point'
        )   

        # ----------------------------------------
        # LABELS
        # ----------------------------------------  
        ax.set_title(
            "3D Terrain Reconstruction",
            fontsize=18,
            fontweight='bold'
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Elevation")

        ax.legend() 

        fig.colorbar(
            surface,
            shrink=0.5,
            aspect=10,
            label='Terrain Elevation'
        )

        # ----------------------------------------
        # SAVE
        # ----------------------------------------
        path = os.path.join(
            self.output_dir,
            "terrain_3d.png"
        )
        plt.savefig(
            path,
            dpi=300
        )

        print(f"🖼️ Saved 3D terrain: {path}")

        plt.show()