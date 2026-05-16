import numpy as np


class LandingSystem:

    def __init__(self, image_shape):

        self.h = image_shape[0]
        self.w = image_shape[1]

    # -------------------------------------------------
    # IMPROVED LANDING SCORE
    # -------------------------------------------------
    def compute_score(self, point, craters):

        distances = []

        for crater in craters.values():

            dist = (
                np.linalg.norm(
                    point - crater.centerpoint
                )
                - crater.diameter / 2
            )

            # INSIDE CRATER = VERY BAD
            if dist < 0:
                return -99999

            distances.append(dist)

        if len(distances) == 0:
            return 0

        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------
        min_clearance = min(distances)

        avg_clearance = np.mean(distances)

        density = np.sum(np.array(distances) < 120)

        # -------------------------------------------------
        # GLOBAL SAFETY SCORE
        # -------------------------------------------------
        score = (
            3.0 * min_clearance
            + 1.5 * avg_clearance
            - 4.0 * density
        )

        return score

    # -------------------------------------------------
    # FIND BEST LANDING POINT
    # -------------------------------------------------
    def find_best_landing_point(self, craters):

        best_score = -1e9

        best_point = None

        # Smaller step = better search
        for y in range(30, self.h - 30, 10):

            for x in range(30, self.w - 30, 10):

                point = np.array([y, x])

                score = self.compute_score(
                    point,
                    craters
                )

                if score > best_score:

                    best_score = score

                    best_point = point

        return best_point, best_score

    # -------------------------------------------------
    # DISTANCE TO NEAREST CRATER
    # -------------------------------------------------
    def distance_to_nearest_crater(
        self,
        point,
        craters
    ):

        distances = []

        for crater in craters.values():

            dist = (
                np.linalg.norm(
                    point - crater.centerpoint
                )
                - crater.diameter / 2
            )

            distances.append(dist)

        return min(distances)

    # -------------------------------------------------
    # HEATMAP
    # -------------------------------------------------
    def generate_heatmap(self, craters):

        heatmap = np.zeros((self.h, self.w))

        for y in range(0, self.h, 4):

            for x in range(0, self.w, 4):

                point = np.array([y, x])

                score = self.compute_score(
                    point,
                    craters
                )

                heatmap[y, x] = score

        return heatmap

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
        by, bx = best_point.astype(int)

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