import cv2
import blur_detector
import multiprocessing

ROOT = '/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/MDE/kitti/'
FILES = 'kitti_eigen_train.txt'
FILES2 = 'kitti_eigen_test.txt'

def convert_img(url):
    img = cv2.imread(ROOT + 'input/' + url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = blur_detector.detectBlur(img, downsampling_factor=4, num_scales=4,
                                    scale_start=2, num_iterations_RF_filter=3)
    res = (blur * 65535).astype('uint16')
    cv2.imwrite(ROOT + 'B/' + url, res)

fls = []
with open(ROOT + FILES, 'r') as f:
    fls += [line.split()[0] for line in f.readlines()]
with open(ROOT + FILES2, 'r') as f:
    fls += [line.split()[0] for line in f.readlines()]

pool = multiprocessing.Pool()
pool.map(convert_img, fls)
