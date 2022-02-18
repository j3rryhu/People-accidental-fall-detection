from pycocotools.coco import COCO
from torch.utils.data import Dataset
from data.data_processing import *
import numpy as np
import math

test_kpt_file = '../COCO dataset/annotations/person_keypoints_val2017.json'


class CocoTrainDataset(Dataset):
    def __init__(self, image_folder, sigma, stride, thickness,paf_thresh, transform=None):
        super(CocoTrainDataset, self).__init__()
        self.train_kpt_file = '../COCO dataset/annotations/person_keypoints_train2017.json'
        self.img_folder = image_folder
        self.sigma = sigma
        self.thickness = thickness
        self.paf_tresh = paf_thresh
        self.stride = stride
        self.transform = transform
        self.BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                      [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
        self.coco = COCO(self.train_kpt_file)
        self.dataset = self.coco.dataset

    def __getitem__(self, idx):
        img = self.coco.loadImgs(idx)
        np.transpose(img, [2, 0, 1])
        ann_id = self.coco.getAnnIds(imgIds=idx)
        annotations_per = self.coco.loadAnns(ann_id)
        kpts = []
        for ann in annotations_per:
            kpts.append(ann['keypoints'])
        for keypoint_group in kpts:
            convert_keypoint(keypoint_group)
        heatmap = self.gen_heatmap(img, kpts)
        paf_map = self.gen_paf(img, kpts)
        return img, heatmap, paf_map

    def __len__(self):
        return len(self.dataset['annotations'])

    def gen_heatmap(self, img, kpts):
        kpt_num = 18
        _, w, h = img.shape()
        heatmap = np.zeros([kpt_num+1, w//self.stride, h//self.stride])
        for c in range(0, kpt_num):
            if kpts[c][2]==1:
                self.gaussian(heatmap[c], kpts[c])
            else:
                continue
        heatmap[-1] = 1 - heatmap.max(axis=0)
        return heatmap

    def gen_paf(self, img, kpt):
        _, w, h = img.shape()
        paf_num = len(self.BODY_PARTS_KPT_IDS)
        pafmap = np.zeros([2*paf_num, w // self.stride, h // self.stride])
        mapped_w = w//self.stride
        mapped_h = h//self.stride
        for c in range(0, 2*paf_num, 2):
            body_connection = self.BODY_PARTS_KPT_IDS[c//2]
            kpt_a = kpt[body_connection[0]]
            kpt_b = kpt[body_connection[1]]
            x1, y1, a1 = kpt_a
            x2, y2, a2 = kpt_b
            if a1==0 or a2==0:
                continue
            x1/=self.stride
            x2/=self.stride
            y1/=self.stride
            y2/=self.stride
            x12 = x2-x1
            y12 = y2-y1
            veclen = (x12**2+y12**2)**0.5
            if veclen<0.0001:
                continue
            x12/=veclen
            y12/=veclen
            x_min = int(max(min(x1, x2) - self.thickness, 0))
            x_max = int(min(max(x1, x2) + self.thickness, mapped_w))
            y_min = int(max(min(y1, y2) - self.thickness, 0))
            y_max = int(min(max(y1, y2) + self.thickness, mapped_h))
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    x_1m = x - x1
                    y_1m = y - y1
                    d = math.fabs(x_1m * y12 - y_1m * x12)
                    parallel = x_1m*x12+y_1m*y12
                    if d <= self.thickness and parallel>0:
                        pafmap[c, y, x] = x12
                        pafmap[c+1, y, x] = y12
        return pafmap



    def gaussian(self, heatmap, kpt):
        _, mapped_w, mapped_h = heatmap.shape()
        x, y, a = kpt
        mapped_x = x / self.stride
        mapped_y = y / self.stride
        for i in range(0, mapped_w):
            for j in range(0, mapped_h):
                gaussian_val = math.exp(-((i-mapped_x)**2+(j-mapped_y)**2)/self.sigma/self.sigma)
                heatmap[i][j] = max(heatmap[i][j], gaussian_val)




