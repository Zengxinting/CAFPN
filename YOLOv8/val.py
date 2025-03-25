from ultralytics import YOLO,RTDETR

model = YOLO("/media/zxt/文档/PycharmProject/2detection/Light-YOLO/ultralytics-main/runs/detect/yolov8cafpn/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(batch=16)  # no arguments needed, dataset and settings remembered8
metrics.box.map  # map50-95UserWarning: adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'.
metrics.box.map50  # map50
metrics.box.maps  # a list contains map50-95 of each category
print(metrics.box.maps)