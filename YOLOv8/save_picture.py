from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("/media/zxt/文档/PycharmProject/2detection/Light-YOLO/ultralytics-main/runs/yolov8/weights/best.pt")
# model = YOLO("/media/zxt/文档/PycharmProject/2detection/Light-YOLO (2)/results/VEDAI-YOLO/yolov9m/weights/best.pt")
# model = YOLO("/media/zxt/文档/PycharmProject/2detection/Light-YOLO (2)/results/VEDAI-YOLO/yolov5m/weights/best.pt")
results = model.predict(source="/media/zxt/文档/PycharmProject/voc/vocYOLO/voctrain_07/images/val/007979.jpg", save=True, save_txt=True,show_labels = True)


# Speed: 1.3ms preprocess, 8.4ms inference, 1.8ms postprocess per image at shape (1, 3, 640, 640) light-yolo
# Speed: 1.4ms preprocess, 6.8ms inference, 1.9ms postprocess per image at shape (1, 3, 640, 640) f-light-yolo11n
#Speed: 1.3ms preprocess, 4.3ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 640) v5
# Speed: 1.3ms preprocess, 5.3ms inference, 1.8ms postprocess per image at shape (1, 3, 640, 640) v8
#Speed: 1.3ms preprocess, 5.8ms inference, 1.8ms postprocess per image at shape (1, 3, 640, 640) v9
# Speed: 1.3ms preprocess, 5.8ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 640) v10