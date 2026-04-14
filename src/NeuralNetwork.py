from ultralytics import YOLO
import numpy as np
from src.Crater import Crater

# Load trained model
model = YOLO("best.pt")


def remove_close_craters(craters):
    filtered = {}
    for k1, c1 in craters.items():
        keep = True
        for k2, c2 in craters.items():
            if k1 == k2:
                continue

            dist = np.linalg.norm(c1.centerpoint - c2.centerpoint)

            if dist < min(c1.diameter, c2.diameter) * 0.5:
                if getattr(c1, "score", 0) < getattr(c2, "score", 0):
                    keep = False
                    break

        if keep:
            filtered[k1] = c1

    return filtered


def estimate_depth(diameter):
    # 🔥 Improved depth model
    return 0.15 * diameter + 0.02 * np.sqrt(diameter)


def detectCratersNN(im):
    img = np.array(im)
    results = model(img)

    craters = {}
    count = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            # 🔥 Strong filtering
            if score < 0.7:
                continue

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 🔥 Better diameter estimation
            diameter = np.median([(x2 - x1), (y2 - y1)])

            depth = estimate_depth(diameter)

            crater = Crater(count, np.array([cy, cx]), diameter)

            crater.depth = depth
            crater.score = float(score)

            craters[count] = crater
            count += 1

    print(f"🧠 NN detected craters: {len(craters)}")

    # Remove overlaps
    craters = remove_close_craters(craters)

    # 🔥 Keep only BEST craters
    craters = dict(
        sorted(craters.items(), key=lambda x: x[1].score, reverse=True)[:15]
    )

    print(f"🧠 NN filtered craters: {len(craters)}")

    return craters