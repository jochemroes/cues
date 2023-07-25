import numpy as np
import torch
import cv2
import PIL
import opt
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import detection
from glob import glob

from net import MADFNet
from util.io import load_ckpt
from util.image import unnormalize

ITS_THRESH = .85
GROW_THRESH = 1.05
SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
obj_model = detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT', progress=True, num_classes=91).to(device)
obj_model.eval()
pnt_model = MADFNet(layer_size=7).to(device)
load_ckpt('saves/places2.pth', [('pnt_model', pnt_model)])
pnt_model.eval()
img_transform = transforms.Compose(
    [transforms.Resize(size=512), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=512, interpolation=PIL.Image.NEAREST), transforms.ToTensor()])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

ROOT = '/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/MDE/kitti/'
FILES = 'kitti_eigen_train.txt'


class Vertex:

    def __init__(self, num):
        self.id = num
        self.children = []
        self.depth = 0

    def add_child(self, v_i):
        self.children.append(v_i)
        for v_j in self.children:
            v_j.update(self.depth)

    def offspring(self, visited):
        for v in self.children:
            if v not in visited:
                visited.add(v)
                visited = v.offspring(visited)
        return visited

    def update(self, depth):
        self.depth = max(self.depth, depth + 1)
        for v in self.children:
            v.update(self.depth)


def box_overlap(x1, y1, x2, y2, p1, q1, p2, q2):
    width = min(x2, p2) - max(x1, p1)
    height = min(y2, q2) - max(y1, q1)
    return width > 0 and height > 0


def get_range(len_in, len_out, pnt):
    if pnt - len_out // 2 < 0 and pnt + len_out // 2 > len_in:
        return 0, len_in
    if pnt - len_out // 2 < 0:
        return 0, len_out
    if pnt + len_out // 2 > len_in:
        return len_in - len_out, len_in
    return pnt - len_out // 2, pnt + len_out // 2


def crop(img, mask, x1, y1, x2, y2, size=SIZE):
    cx, cy = (x2 + x1) // 2, (y2 - y1) // 2
    w1, w2 = get_range(img.shape[1], size, cx)
    h1, h2 = get_range(img.shape[0], size, cy)
    cropped = img[int(h1):int(h2), int(w1):int(w2)]
    res_img = np.zeros((size, size, 3), dtype='uint8')
    maybe_x = 0 if cropped.shape[1] % 2 == 0 else 1
    maybe_y = 0 if cropped.shape[0] % 2 == 0 else 1
    res_img[size // 2 - cropped.shape[0] // 2:size // 2 + cropped.shape[0] // 2 + maybe_y,
            size // 2 - cropped.shape[1] // 2:size // 2 + cropped.shape[1] // 2 + maybe_x] = cropped
    crop_mask = (mask[int(h1):int(h2), int(w1):int(w2)]).astype('float32')
    dilate_mask = cv2.dilate(crop_mask, kernel, iterations=1)
    res_mask = np.zeros((size, size), dtype='float32')
    res_mask[size // 2 - cropped.shape[0] // 2:size // 2 + cropped.shape[0] // 2 + maybe_y,
             size // 2 - cropped.shape[1] // 2:size // 2 + cropped.shape[1] // 2 + maybe_x] = dilate_mask
    res_mask = 1 - res_mask
    return res_img, res_mask, *cropped.shape[:2], w1, h1


def paint_in(img, cropped, mask, height, width, x, y):
    inp = img_transform(PIL.Image.fromarray(cropped)).to(device)
    mask = mask_transform(PIL.Image.fromarray(mask)).to(device)
    mask = torch.cat([mask, mask, mask])
    inp = inp * mask
    inp = torch.stack((inp,))
    mask = torch.stack((mask,))
    with torch.no_grad():
        outputs = pnt_model(inp, mask)
    inpnt = mask * inp + (1 - mask) * outputs[-1]
    inpnt = unnormalize(inpnt.cpu()) * 255
    inpnt = inpnt.numpy()[0].transpose((1, 2, 0))
    maybe_x = 0 if width % 2 == 0 else 1
    maybe_y = 0 if height % 2 == 0 else 1
    inpnt = inpnt[inpnt.shape[0] // 2 - height // 2: inpnt.shape[0] // 2 + height // 2 + maybe_y,
                  inpnt.shape[1] // 2 - width // 2: inpnt.shape[1] // 2 + width // 2 + maybe_x]
    res = np.copy(img)
    res[int(y):int(y) + height, int(x):int(x) + width] = inpnt
    return res


def detect(img):
    inp = np.expand_dims(img.transpose((2, 0, 1)) / 255, axis=0)
    detections = obj_model(torch.FloatTensor(inp).to(device))[0]
    return [(mask[0], box, mask[0].sum()) for mask, box, conf in zip(
        detections['masks'].detach().cpu().numpy() > .5,
        detections['boxes'].detach().cpu().numpy(),
        detections['scores'].detach().cpu().numpy()) if conf > .5]


def growth(obj, img):
    mask1, _, cnt = obj
    detections = detect(img)
    for mask2, _, cnt2 in detections:
        mask = np.zeros(mask1.shape, dtype='bool')
        mask[mask1 & mask2] = True
        if mask.sum() / cnt > ITS_THRESH:
            return cnt2 / cnt
    # no match
    return 0


def overlapping(img):
    objs = detect(img)
    i = 0
    while i < len(objs):
        mask1, _, cnt = objs[i]
        j = i + 1
        while j < len(objs):
            mask2 = objs[j][0]
            mask = np.zeros(mask1.shape)
            mask[mask1 & mask2] = True
            if mask.sum() / cnt > ITS_THRESH:
                objs.pop(j)
            j += 1
        i += 1
    pairs = []
    for i, (_, box1, _) in enumerate(objs):
        for delta, (_, box2, _) in enumerate(objs[i + 1:]):
            j = i + 1 + delta
            if box_overlap(*box1, *box2):
                pairs.append((i, j))
    inpainted = [None] * len(objs)
    graph = [Vertex(i) for i in range(len(objs))]
    for i, j in pairs:
        if inpainted[i] is None:
            inpainted[i] = paint_in(img, *crop(img, objs[i][0], *objs[i][1]))
        if inpainted[j] is None:
            inpainted[j] = paint_in(img, *crop(img, objs[j][0], *objs[j][1]))
        growth_i, growth_j = growth(objs[i], inpainted[j]), growth(objs[j], inpainted[i])
        if growth_i > 0 or growth_j > 0:
            edge = None
            if growth_i > growth_j and growth_i > GROW_THRESH:  # obj j is closer than obj i
                edge = j, i
            elif growth_j > growth_i and growth_j > GROW_THRESH:  # obj i is closer than obj j
                edge = i, j
            if edge is not None and graph[edge[0]] not in graph[edge[1]].offspring(set()):
                graph[edge[0]].add_child(graph[edge[1]])
    return objs, graph


def make_image(url):
    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    objs, graph = overlapping(img)
    res = np.zeros(img.shape[:2], dtype='uint8')
    for i, (mask, _, _) in enumerate(objs):
        res[mask] = graph[i].depth + 1
    return res

with open(ROOT + FILES, 'r') as f:
    for line in f.readlines()[::-1]:
        url = line.split()[0]
        inp = ROOT + 'input/' + url
        outp = ROOT + 'O/' + url
        overlap = make_image(inp)
        cv2.imwrite(outp, overlap)
