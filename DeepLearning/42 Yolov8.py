##############################
#### 39 YOLOV8 INFERENCIA ########
###############################

from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor as s
from ultralytics.yolo.v8.segment.predict import DetectionPredictor
import cv2
model = YOLO("modelos/YOLO8/yolo/yolov8x.pt")
# results = model.predict(source="modelos/YOLO8/1.mov", show=True)  # acepta todos los formatos - img/carpeta/video
results = model.predict(source=4, show=True)  # acepta todos los formatos - img/carpeta/video
print(results)
results.get