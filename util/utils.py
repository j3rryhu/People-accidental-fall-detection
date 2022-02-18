import cv2
import numpy as np



def extract_keypoints(heatmap, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    c, w, h = heatmap.shape
    kpts = []
    for idx in range(0, c):
        one_kpt_group = []
        one_heatmap = heatmap[idx, :, :]
        smth_heatmap = cv2.GaussianBlur(one_heatmap, (5, 5), 0)
        smth_heatmap = np.uint8(smth_heatmap>threshold)
        _, contours, _ = cv2.findContours(smth_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        id = 0
        for cnt in contours:
            mask = np.zeros(smth_heatmap.shape(), dtype=np.uint8)
            mask = cv2.drawContours(mask, cnt, -1, 1)
            segment = smth_heatmap*mask
            _, maxval, _, maxloc = cv2.minMaxLoc(segment)
            y, x = maxloc
            kpt = (y, x, maxval, id)
            one_kpt_group.append(kpt)
            id+=1
        kpts.append(one_kpt_group)
    np.vstack(kpts)
    np.squeeze(kpts)
    return kpts


def calc_paf_score(k, pafmap, kpts):
    pafmap = np.squeeze(pafmap)
    POSE_PAIR = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                 [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                 [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                 [2, 17], [5, 16]]
    kptA = kpts[POSE_PAIR[k][0], :]
    kptB = kpts[POSE_PAIR[k][1], :]



def connect_kpt(pafmap):
    score_thresh = 0.7
