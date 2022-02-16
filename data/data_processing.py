def convert_keypoint(kpt):
    reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    packed_kpt = []
    for idx in reorder_map:
        idx -= 1
        x = kpt[3*idx]
        y = kpt[3*idx+1]
        a = kpt[3*idx+2]
        if a == 2:
            a = 1
        ann = [x,y,a]
        packed_kpt.append(ann)
    if packed_kpt[5][2]!=0 and packed_kpt[6][2]!=0:
        packed_kpt.insert(1, [(packed_kpt[5][0] + packed_kpt[6][0]) / 2, (packed_kpt[5][1] + packed_kpt[6][1]) / 2, 0])
    else:
        packed_kpt.insert([0,0,0])
    return packed_kpt
