# src/LandingSystem.py

import numpy as np


class LandingSystem:

    def __init__(self, image_shape):

        self.height = image_shape[0]
        self.width  = image_shape[1]

        # Pre-compute normalisation constants so every term
        # lives in roughly the same numeric range (~0-1).
        self._max_dist = np.sqrt(self.height ** 2 + self.width ** 2)
        self._max_dim  = max(self.height, self.width)

    # ----------------------------------------
    # DISTANCE TO NEAREST CRATER EDGE
    # ----------------------------------------
    def distance_to_nearest_crater(self, point, craters):
        """Returns clearance from point to the nearest crater rim (px)."""
        py, px = point
        min_dist = float('inf')

        for crater in craters.values():
            cy, cx   = crater.centerpoint
            dist     = np.linalg.norm(np.array([py, px]) - np.array([cy, cx]))
            edge_dist = dist - crater.diameter / 2
            if edge_dist < min_dist:
                min_dist = edge_dist

        return max(min_dist, 0.0)

    # ----------------------------------------
    # LOCAL CRATER DENSITY
    # ----------------------------------------
    def crater_density(self, point, craters, radius=80):
        """Number of crater centres within `radius` px of point."""
        py, px  = point
        density = 0

        for crater in craters.values():
            cy, cx = crater.centerpoint
            dist   = np.linalg.norm(np.array([py, px]) - np.array([cy, cx]))
            if dist < radius:
                density += 1

        return density

    # ----------------------------------------
    # TERRAIN SMOOTHNESS
    # depth-weighted influence of all craters
    # ----------------------------------------
    def terrain_smoothness(self, point, craters):
        """Depth-weighted crater influence (roughness proxy)."""
        py, px     = point
        smoothness = 0.0

        for crater in craters.values():
            cy, cx = crater.centerpoint
            dist   = np.linalg.norm(np.array([py, px]) - np.array([cy, cx]))
            smoothness += crater.depth / (dist + 1.0)

        return smoothness

    # ----------------------------------------
    # TERRAIN SLOPE  (finite-difference on depth field)
    # This is genuinely different from smoothness:
    # it measures the *gradient* of the depth field
    # at the candidate point, not the total influence.
    # ----------------------------------------
    def terrain_slope(self, point, craters, delta=5):
        """
        Estimate local slope via finite differences on the
        reconstructed depth surface at `point`.
        A flat inter-crater plateau → slope ≈ 0.
        A crater rim → slope is large.
        """
        py, px = point

        def depth_at(y, x):
            val = 0.0
            for crater in craters.values():
                cy, cx  = crater.centerpoint
                r       = crater.diameter / 2.0
                dist_sq = (y - cy) ** 2 + (x - cx) ** 2
                # Gaussian depression
                val -= crater.depth * np.exp(-dist_sq / (2.0 * r ** 2))
            return val

        dzdx = (depth_at(py, px + delta) - depth_at(py, px - delta)) / (2 * delta)
        dzdy = (depth_at(py + delta, px) - depth_at(py - delta, px)) / (2 * delta)

        return np.sqrt(dzdx ** 2 + dzdy ** 2)   # gradient magnitude

    # ----------------------------------------
    # BOUNDARY CENTRALITY  (reward, not penalty)
    # Returns 0.0 at image corners, 1.0 at image centre.
    # ----------------------------------------
    def boundary_centrality(self, point):
        """
        True centre-biased reward.
        Peaks at 1.0 at the image centre; falls to 0.0 at the corners.
        This avoids the margin-clipping artifact where points exactly at
        `margin` distance from the edge score identically to the centre.
        """
        py, px = point

        centre_y = self.height / 2.0
        centre_x = self.width  / 2.0

        # Maximum possible distance from centre (corner distance)
        max_dist = np.sqrt(centre_y ** 2 + centre_x ** 2)

        dist = np.sqrt((py - centre_y) ** 2 + (px - centre_x) ** 2)

        return float(1.0 - dist / max_dist)   # 1.0 at centre, 0.0 at corner

    # ----------------------------------------
    # LANDING SCORE  (all terms normalised ~0-1)
    # ----------------------------------------
    def landing_score(self, point, craters):
        """
        Higher = safer to land.

        Positive (rewards):
          min_clearance  – stay away from crater rims
          centrality     – stay away from image borders

        Negative (penalties):
          density        – avoid crowded regions
          smoothness     – avoid deep/close craters
          slope          – avoid steep terrain

        All terms are scaled so they contribute comparably.
        """
        # ---- rewards ----
        min_clearance = self.distance_to_nearest_crater(point, craters)
        norm_clearance = min_clearance / self._max_dim          # 0-1

        centrality = self.boundary_centrality(point)            # 0-1

        # ---- penalties ----
        density    = self.crater_density(point, craters)        # integer, max ~12
        smoothness = self.terrain_smoothness(point, craters)    # depth/px units
        slope      = self.terrain_slope(point, craters)         # unitless gradient

        # Normalise penalties to ~0-1 range
        norm_density    = density    / max(len(craters), 1)
        norm_smoothness = np.clip(smoothness / 5.0, 0.0, 1.0)  # empirical cap
        norm_slope      = np.clip(slope      / 2.0, 0.0, 1.0)  # gradient cap

        score = (
              5.0 * norm_clearance    # primary safety driver
            + 3.0 * centrality        # penalise edges gently
            - 4.0 * norm_density      # avoid crater clusters
            - 2.0 * norm_smoothness   # avoid rough terrain
            - 2.0 * norm_slope        # avoid steep slopes
        )

        return score

    # ----------------------------------------
    # FIND BEST LANDING POINT  (coarse → fine)
    # ----------------------------------------
    def find_best_landing_point(self, craters, margin=40):
        """
        Two-pass coarse-to-fine search so we don't miss the true optimum.

        Pass 1: coarse grid (step=40) to locate the best region.
        Pass 2: fine grid (step=10) within 60 px of the coarse best.
        """
        # Pass 1 – coarse
        best_score  = -float('inf')
        best_coarse = None

        for y in range(margin, self.height - margin, 40):
            for x in range(margin, self.width - margin, 40):
                s = self.landing_score((y, x), craters)
                if s > best_score:
                    best_score  = s
                    best_coarse = (y, x)

        # Pass 2 – fine, within a window around the coarse winner
        cy, cx = best_coarse
        window = 60

        for y in range(max(margin, cy - window),
                       min(self.height - margin, cy + window), 10):
            for x in range(max(margin, cx - window),
                           min(self.width  - margin, cx + window), 10):
                s = self.landing_score((y, x), craters)
                if s > best_score:
                    best_score = s
                    best_coarse = (y, x)

        best_point = np.array(best_coarse).astype(int)
        return best_point, best_score

    # ----------------------------------------
    # HEATMAP
    # ----------------------------------------
    def generate_heatmap(self, craters, step=10):
        """Score every grid cell for visualisation."""
        heatmap = np.zeros((self.height, self.width))

        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                score = self.landing_score((y, x), craters)
                heatmap[y:y + step, x:x + step] = score

        return heatmap