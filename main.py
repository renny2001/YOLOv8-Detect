# pip uninstall pip setuptools
# pip3 install --upgrade pip
# pip3 install -upgrade setuptools
# pip3 install torch torchvision torchaudio--index-url https://download. pytorch.org/whl/cu118
# git clone https://github.com/ultralytics/ultralytics/ultralytics
# pip install ultralytics

import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO(' models/yolov8s.pt')
    model.to('cuda')

    results = model.train(
        data="coffee.yaml",
        workers=1,
        device=0,
        imgsz=256,
        epochs=300,
        patience=50,
        batch=-1,
        project='YOLOv8',
        name='exp01;')
