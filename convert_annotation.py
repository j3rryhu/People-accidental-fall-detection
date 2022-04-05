import json
from pycocotools.coco import COCO
coco_reader = COCO('./COCO dataset/annotations/person_keypoints_train2017.json')
img_ids = coco_reader.getImgIds()
json_dict = {}
count = 0
for id in img_ids:
    ann_id = coco_reader.getAnnIds(id)
    ann = coco_reader.loadAnns(ann_id)
    for anno in ann:
        json_dict[count] = anno
        count += 1
with open('./train_annotation.json', 'w') as f:
    json.dump(json_dict, f)
print('{} annotations are uploaded'.format(len(json_dict)))