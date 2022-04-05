'''
 loadImgs is used to get the information on the image
 id means the specified number for each image
'''
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from data.data_processing import *
from torchvision import transforms
import numpy as np
import math
from PIL import Image
import os
import json

test_kpt_file = '../COCO dataset/annotations/person_keypoints_val2017.json'


class CocoTrainDataset(Dataset):
    def __init__(self, image_folder, sigma, stride, thickness, transform=None):
        super(CocoTrainDataset, self).__init__()
        self.train_kpt_file = './train_annotation.json'
        self.original_file = './COCO dataset/annotations/person_keypoints_train2017.json'
        self.img_folder = image_folder
        self.sigma = sigma
        self.thickness = thickness
        self.stride = stride
        self.transform = transform
        self.BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                      [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
        self.coco = COCO(self.original_file)
        self.ids = self.coco.getImgIds()
        with open(self.train_kpt_file, 'r') as f:
            self.ann_dict = json.load(f)
        self.size = (75, 100)

    def __getitem__(self, idx):
        img_id = self.ann_dict[str(idx)]['image_id']
        info = self.coco.loadImgs(img_id)
        img_name = info[0]['file_name']
        img = Image.open(os.path.join(self.img_folder, 'train2017', img_name))
        img = img.convert('RGB')
        kpts = []
        for i in range(0, len(self.ann_dict[str(idx)]['keypoints'])):
            kpts.append(self.ann_dict[str(idx)]['keypoints'][i])
        kpts = convert_keypoint(kpts)
        w, h = img.size
        x_scale = 150 / w
        y_scale = 142 / h
        for kpt in kpts:
            kpt[0] *= x_scale
            kpt[1] *= y_scale
        img = img.resize((150, 142))
        img = self.transform(img)
        heatmap = self.gen_heatmap(img, kpts)
        paf_map = self.gen_paf(img, kpts)
        sample = {
            'image': img,
            'heatmap': heatmap,
            'paf_map': paf_map
        }
        return sample

    def __len__(self):
        return len(self.ann_dict)

    def gen_heatmap(self, img, kpts):
        kpt_num = 18
        _, h, w = img.shape
        heatmap = np.zeros([kpt_num+1, h//self.stride, w//self.stride])
        for c in range(0, kpt_num):
            if kpts[c][2]==1:
                self.gaussian(heatmap[c], kpts[c])
            else:
                continue
        heatmap[-1] = 1 - heatmap.max(axis=0)
        return heatmap

    def gen_paf(self, img, kpt):
        _, h, w = img.shape
        paf_num = len(self.BODY_PARTS_KPT_IDS)
        pafmap = np.zeros([2*paf_num, h // self.stride, w // self.stride])
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
        mapped_h, mapped_w = heatmap.shape
        x, y, a = kpt
        mapped_x = x / self.stride
        mapped_y = y / self.stride
        for i in range(0, mapped_h):
            for j in range(0, mapped_w):
                gaussian_val = math.exp(-((i-mapped_x)**2+(j-mapped_y)**2)/self.sigma/self.sigma)
                heatmap[i][j] = max(heatmap[i][j], gaussian_val)




