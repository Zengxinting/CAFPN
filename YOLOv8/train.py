from ultralytics import YOLO,RTDETR

model = YOLO("/media/zxt/文档/PycharmProject/2detection/Light-YOLO/yolo11.yaml")
# model.load("/media/zxt/文档/PycharmProject/2detection/Light-YOLO/yolov8-p6.yaml")
# print(model)
model.train(
    data="/media/zxt/文档/PycharmProject/2detection/Light-YOLO/ultralytics-main/ultralytics/cfg/datasets/VOC.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    # resume=True,
    batch=16,
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # nwdloss= False,
    pretrained=True,
)



# yolo train resume model=/media/zxt/文档/PycharmProject/2detection/yolo11/ultralytics-main/runs/detect/train/weights/best.pt


