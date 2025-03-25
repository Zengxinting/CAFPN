# import os
# import shutil
# import xml.etree.ElementTree as ET
#
# # VOC格式数据集路径
# voc_data_path = '/media/zxt/文档/PycharmProject/voc/VOCdevkit/VOC2012'
# voc_annotations_path = os.path.join(voc_data_path, 'Annotations')
# voc_images_path = os.path.join(voc_data_path, 'JPEGImages')
#
# # YOLO格式数据集保存路径
# yolo_data_path = '/media/zxt/文档/PycharmProject/voc/vocYOLO'
# yolo_images_path = os.path.join(yolo_data_path, 'images1')
# yolo_labels_path = os.path.join(yolo_data_path, 'labels1')
#
# # 创建YOLO格式数据集目录
# os.makedirs(yolo_images_path, exist_ok=True)
# os.makedirs(yolo_labels_path, exist_ok=True)
#
# # 类别映射 (可以根据自己的数据集进行调整)
# class_mapping = {
#     'aeroplane':0,
#     'bicycle':1,
#     'bird':2,
#     'boat':3,
#     'bottle':4,
#     'bus':5,
#     'car':6,
#     'cat':7,
#     'chair':8,
#     'cow':9,
#     'diningtable':10,
#     'dog':11,
#     'horse':12,
#     'motorbike':13,
#     'person':14,
#     'pottedplant':15,
#     'sheep':16,
#     'sofa':17,
#     'train':18,
#     'tvmonitor':19
# }
#
# def convert_voc_to_yolo(voc_annotation_file, yolo_label_file):
#     tree = ET.parse(voc_annotation_file)
#     root = tree.getroot()
#
#     size = root.find('size')
#     width = float(size.find('width').text)
#     height = float(size.find('height').text)
#
#     with open(yolo_label_file, 'w') as f:
#         for obj in root.findall('object'):
#             cls = obj.find('name').text
#             if cls not in class_mapping:
#                 continue
#             cls_id = class_mapping[cls]
#             xmlbox = obj.find('bndbox')
#             xmin = float(xmlbox.find('xmin').text)
#             ymin = float(xmlbox.find('ymin').text)
#             xmax = float(xmlbox.find('xmax').text)
#             ymax = float(xmlbox.find('ymax').text)
#
#             x_center = (xmin + xmax) / 2.0 / width
#             y_center = (ymin + ymax) / 2.0 / height
#             w = (xmax - xmin) / width
#             h = (ymax - ymin) / height
#
#             f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")
#
# # 遍历VOC数据集的Annotations目录，进行转换
# for voc_annotation in os.listdir(voc_annotations_path):
#     if voc_annotation.endswith('.xml'):
#         voc_annotation_file = os.path.join(voc_annotations_path, voc_annotation)
#         image_id = os.path.splitext(voc_annotation)[0]
#         voc_image_file = os.path.join(voc_images_path, f"{image_id}.jpg")
#         yolo_label_file = os.path.join(yolo_labels_path, f"{image_id}.txt")
#         yolo_image_file = os.path.join(yolo_images_path, f"{image_id}.jpg")
#
#         convert_voc_to_yolo(voc_annotation_file, yolo_label_file)
#         if os.path.exists(voc_image_file):
#             shutil.copy(voc_image_file, yolo_image_file)
#
# print("转换完成！")


import os
import shutil
import random

def make_yolo_dataset(images_folder, labels_folder, output_folder, train_ratio=0.8):
    # 创建目标文件夹
    images_train_folder = os.path.join(output_folder, 'images/train')
    images_val_folder = os.path.join(output_folder, 'images/val')
    labels_train_folder = os.path.join(output_folder, 'labels/train')
    labels_val_folder = os.path.join(output_folder, 'labels/val')

    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(labels_val_folder, exist_ok=True)

    # 获取图片和标签的文件名（不包含扩展名）
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    image_base_names = set(os.path.splitext(f)[0] for f in image_files)
    label_base_names = set(os.path.splitext(f)[0] for f in label_files)

    # 找出图片和标签都存在的文件名
    matched_files = list(image_base_names & label_base_names)

    # 打乱顺序并划分为训练集和验证集
    random.shuffle(matched_files)
    split_idx = int(len(matched_files) * train_ratio)
    train_files = matched_files[:split_idx]
    val_files = matched_files[split_idx:]

    # 移动文件到对应文件夹
    for base_name in train_files:
        img_src = os.path.join(images_folder, f"{base_name}.jpg")
        lbl_src = os.path.join(labels_folder, f"{base_name}.txt")

        img_dst = os.path.join(images_train_folder, f"{base_name}.jpg")
        lbl_dst = os.path.join(labels_train_folder, f"{base_name}.txt")

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(lbl_src, lbl_dst)

    for base_name in val_files:
        img_src = os.path.join(images_folder, f"{base_name}.jpg")
        lbl_src = os.path.join(labels_folder, f"{base_name}.txt")

        img_dst = os.path.join(images_val_folder, f"{base_name}.jpg")
        lbl_dst = os.path.join(labels_val_folder, f"{base_name}.txt")

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(lbl_src, lbl_dst)

    print("数据集划分完成！")

# 使用示例
images_folder = '/media/zxt/文档/PycharmProject/voc/vocYOLO/voc2012/images'  # 原始图片文件夹路径
labels_folder = '/media/zxt/文档/PycharmProject/voc/vocYOLO/voc2012/labels'  # 原始标签文件夹路径
output_folder = '/media/zxt/文档/PycharmProject/voc/vocYOLO/voctrain_12'  # 存放结果数据集的文件夹路径
make_yolo_dataset(images_folder, labels_folder, output_folder)

