import cv2
import numpy as np


def lin_integral(l, pti, ptj):
    dij = np.subtract(pti, ptj)
    veclen = np.linalg.norm(dij)
    if veclen:
        dij/=veclen
        score = np.dot(l, dij)
        return score
    else:
        return 'no score'


def extract_keypoints(heatmap, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    c, w, h = heatmap.shape
    kpts = []
    kpt_list = []
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
            kpt = (x, y, maxval, id)
            one_kpt_group.append(kpt)
            kpt_list.append(kpt)
            id+=1
        kpts.append(one_kpt_group)
    np.vstack(kpts)
    np.squeeze(kpts)
    return kpts, kpt_list


def connect_paf(pafmap, kpts, framewidth, frameheight):
    pafmap = np.squeeze(pafmap)
    valid_pairs = []
    invalid_pairs = []
    POSE_PAIR = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                 [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                 [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                 [2, 17], [5, 16]]
    n_interp_samples = 10
    channels, _, _ = pafmap.shape
    for k in range(0, channels):
        kptA = kpts[POSE_PAIR[k][0], :]
        kptB = kpts[POSE_PAIR[k][1], :]
        pafx = pafmap[2*k, :, :]
        pafy = pafmap[2*k+1, :, :]
        pafx = np.resize(pafx, [framewidth, frameheight])
        pafy = np.resize(pafy, [framewidth, frameheight])
        paf_score_th = 0.1
        conf_th = 0.7
        if len(kptA)!=0 and len(kptB)!=0:
            valid_pair = np.zeros((0, 3))
            for i in range(0, len(kptA)):
                max_score = 0
                max_j = 0
                found = 0
                for j in range(0, len(kptB)):
                    pti = np.array(kptA[:2])
                    ptj = np.array(kptB[:2])
                    x_start = pti[0]
                    y_start = pti[1]
                    x_end = ptj[0]
                    y_end = ptj[1]
                    x_coords = np.linspace(x_start, x_end, num=n_interp_samples)
                    y_coords = np.linspace(y_start, y_end, num=n_interp_samples)
                    scores = []
                    for idx in range(0, len(x_coords)):
                        l = np.array([pafx[y_coords[idx], x_coords[idx]], pafy[y_coords[idx], x_coords[idx]]])
                        if lin_integral(l, pti, ptj) == 'no score':
                            continue
                        scores.append(lin_integral(l, pti, ptj))
                    avg_score = sum(scores)/len(scores)
                    scores = np.array(scores)
                    if (len(np.where(scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if (avg_score > max_score):
                            max_j = j
                            max_score = avg_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[kptA[i][3], kptB[max_j][3], max_score]],
                                           axis=0)
            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def store_person_keypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwiseKeypoints = -1*np.ones((0,19))
    POSE_PAIR = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                 [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                 [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                 [2, 17], [5, 16]]
    for k in range(0, len(POSE_PAIR)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = POSE_PAIR[k]
            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break
                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][
                        2]
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints
