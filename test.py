import ultralytics
ultralytics.checks()

from ultralytics import YOLO
model = YOLO("models/best.pt")
model.predict(
    source="images/train/images",
    conf=0.25,
    save_txt=True,
    save_conf=False,
    save_crop=True,
    visualize=False,
    save=True,
    device=0
)