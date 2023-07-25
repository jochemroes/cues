import numpy as np
import torch
import cv2

from torchvision.models import detection
from glob import glob

ROOT = '/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/MDE/kitti/'
FILES = 'kitti_eigen_test.txt'

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# UNIT CM, HxWXD based on 3D bounding box for object facing away from camera, pretty rough estimation
size_map = {
    'BG': (0, 0, 0),
    'person': (170, 42, 20),
    'bicycle': (105, 40, 175),
    'car': (145, 175, 410),
    'motorcycle': (107, 90, 230),
    # 'airplane' : (1920, 5943, 7041), TOO BIG FOR STORAGE AND NOT VERY USEFUL ANYWAY
    'bus': (299, 255, 1195),
    # 'train' : (430, 300, 2070), based on single carriage but still TOO BIG
    'truck': (200, 240, 520),
    # 'boat' : (1100, 1100, 5000),
    'traffic light': (150, 50, 20),
    'fire hydrant': (60, 20, 20),
    'stop sign': (70, 70, 3),
    'parking meter': (200, 30, 30),
    'bench': (80, 50, 250),
    'bird': (7, 20, 20),
    'cat': (30, 15, 50),
    'dog': (60, 30, 80),
    'horse': (150, 60, 250),
    'sheep': (60, 30, 80),
    'cow': (150, 60, 250),
    'elephant': (200, 150, 300),
    'bear': (150, 100, 150),
    'zebra': (150, 60, 250),
    'giraffe': (400, 60, 250),
    'backpack': (40, 30, 15),
    'umbrella': (100, 100, 100),
    'handbag': (30, 40, 15),
    'tie': (50, 5, 0),
    'suitcase': (30, 40, 15),
    'frisbee': (3, 20, 20),
    'skis': (3, 20, 160),
    'snowboard': (3, 20, 160),
    'sports ball': (20, 20, 20),
    'kite': (150, 50, 0),
    'baseball bat': (5, 5, 30),
    'baseball glove': (3, 10, 20),
    'skateboard': (10, 15, 40),
    'surfboard': (7, 25, 200),
    'tennis racket': (5, 20, 60),
    'bottle': (7, 7, 20),
    'wine glass': (5, 15, 5),
    'cup': (7, 7, 10),
    'fork': (3, 3, 12),
    'knife': (3, 3, 12),
    'spoon': (3, 3, 12),
    'bowl': (7, 12, 12),
    'banana': (3, 5, 12),
    'apple': (7, 7, 7),
    'sandwich': (3, 10, 10),
    'orange': (9, 9, 9),
    'broccoli': (12, 12, 12),
    'carrot': (3, 3, 10),
    'hot dog': (5, 5, 12),
    'pizza': (3, 22, 22),
    'donut': (4, 8, 8),
    'cake': (8, 18, 18),
    'chair': (100, 50, 50),
    'couch': (100, 100, 300),
    'potted plant': (30, 15, 15),
    'bed': (80, 160, 220),
    'dining table': (100, 150, 300),
    'toilet': (70, 30, 50),
    'tv': (70, 120, 7),
    'laptop': (20, 30, 20),
    'mouse': (3, 5, 8),
    'remote': (3, 5, 20),
    'keyboard': (3, 25, 10),
    'cell phone': (0, 6, 13),
    'microwave': (20, 30, 30),
    'oven': (50, 70, 50),
    'toaster': (12, 12, 25),
    'sink': (15, 20, 15),
    'refrigerator': (200, 50, 30),
    'book': (5, 10, 15),
    'clock': (15, 15, 5),
    'vase': (20, 10, 10),
    'scissors': (1, 5, 12),
    'teddy bear': (20, 15, 15),
    'hair drier': (10, 5, 14),
    'toothbrush': (13, 3, 3)
}

mx = 0
for h, w, d in size_map.values():
    mx = max(mx, h, w, d)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT', progress=True, num_classes=91).to(device)
model.eval()
with open(ROOT + FILES, 'r') as f:
    for line in f.readlines():
        url = line.split()[0]
        inp = ROOT + 'input/' + url
        outp = ROOT + 'F/' + url
        img = cv2.imread(inp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = img.shape
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img / 255
        img = torch.FloatTensor(img).to(device)
        detections = model(img)[0]
        masks = detections['masks'].detach().cpu().numpy() > .5
        labs = detections['labels'].detach().cpu().numpy()
        confs = detections['scores'].detach().cpu().numpy()
        res = np.zeros(shape)
        for mask, lab, conf in zip(masks, labs, confs):
            if lab < 81 and class_names[lab] in size_map and conf > .5:
                res[mask[0]] = size_map[class_names[lab]]
        res = (res / mx * 255).astype('uint8')
        res_url = outp
        cv2.imwrite(res_url, res)
