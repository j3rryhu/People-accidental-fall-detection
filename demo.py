from util.utils import *
from model.recursive_model import BodyPoseModel
import cv2
import torch


ckpt_file = torch.load('./model/280_trained_model.pth')
model = BodyPoseModel()
model.load_state_dict(ckpt_file['net'])
video_pth = ' ' # fill in the video path
video = cv2.VideoCapture(video_pth)
while True:
    ret, frame = video.read()
    out1, out2 = model(frame)
    _, w, h = frame.shape()
    kpts, kpt_list = extract_keypoints(out1)
    validpair, invalidpair = connect_paf(out2, kpts, w, h)
    POSE_PAIR = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                 [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                 [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                 [2, 17], [5, 16]]
    kpt_per_person = store_person_keypoints(validpair, invalidpair, kpt_list)
    for personidx in range(len(kpt_per_person)):
        pos = personidx%3
        color = [0,0,0]
        color[pos] = 255
        for i in range(17):
            pt_idx1 = kpt_per_person[personidx][POSE_PAIR[i][0]]
            pt_idx2 = kpt_per_person[personidx][POSE_PAIR[i][1]]
            pt1 = kpt_list[pt_idx1][:2]
            pt2 = kpt_list[pt_idx2][:2]
            cv2.line(frame, pt1, pt2, color)
    cv2.imshow('frame', frame)