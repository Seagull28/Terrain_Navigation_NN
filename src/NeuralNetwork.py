from ultralytics import YOLO
import numpy as np
from src.Crater import Crater


class NeuralDetector:

    def __init__(self, model_path="best.pt", conf_threshold=0.7):

        self.model = YOLO(model_path)

        self.conf_threshold = conf_threshold

    # ----------------------------------------
    # REMOVE OVERLAPPING CRATERS
    # ----------------------------------------
    def remove_close_craters(self, craters):

        keys = list(craters.keys())

        to_remove = set()

        for i, k1 in enumerate(keys):

            if k1 in to_remove:
                continue

            for k2 in keys[i + 1:]:

                if k2 in to_remove:
                    continue

                c1 = craters[k1]
                c2 = craters[k2]

                dist = np.linalg.norm(
                    c1.centerpoint - c2.centerpoint
                )

                if dist < min(c1.radius, c2.radius):

                    worse = (
                        k1
                        if c1.score < c2.score
                        else k2
                    )

                    to_remove.add(worse)

        return {
            k: v
            for k, v in craters.items()
            if k not in to_remove
        }

    # ----------------------------------------
    # DEPTH ESTIMATION
    # ----------------------------------------
    def estimate_depth(self, diameter):

        return 0.15 * diameter + 0.02 * np.sqrt(diameter)

    # ----------------------------------------
    # DETECTION
    # ----------------------------------------
    def detectCratersNN(self, im):

        img = np.array(im)

        results = self.model(img)

        craters = {}

        count = 0

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()

            scores = r.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):

                if score < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                diameter = np.median([
                    (x2 - x1),
                    (y2 - y1)
                ])

                crater = Crater(
                    count,
                    np.array([cy, cx]),
                    diameter,
                    float(score)
                )

                crater.depth = self.estimate_depth(
                    diameter
                )

                craters[count] = crater

                count += 1

        print(f"🧠 NN detected craters: {len(craters)}")

        craters = self.remove_close_craters(
            craters
        )

        craters = dict(
            sorted(
                craters.items(),
                key=lambda x: x[1].score,
                reverse=True
            )[:15]
        )

        print(f"🧠 NN filtered craters: {len(craters)}")

        return craters