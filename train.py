from ultralytics import YOLO

# 🔥 Use slightly bigger model (better accuracy)
model = YOLO("yolov8s.pt")

model.train(
    data="crater.yaml",
    epochs=40,        # ✅ reduced
    imgsz=512,        # ✅ faster
    batch=8,
    patience=10,      # ✅ early stop
    degrees=5,
    scale=0.3,
    fliplr=0.5,
    mosaic=0.5,
    mixup=0.1
)