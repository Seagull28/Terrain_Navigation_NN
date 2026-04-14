import numpy as np
from PIL import Image, ImageDraw
from src.NeuralNetwork import detectCratersNN
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
        draw.ellipse((middle[0]-5, middle[1]-5, middle[0]+5, middle[1]+5), fill='green')

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

        self.drawDescentImageOnReferenceImage(middle, descentImageCraters)
        self.drawCraterDistances(descentImageCraters)

        print("✅ Localization successful:", middle)