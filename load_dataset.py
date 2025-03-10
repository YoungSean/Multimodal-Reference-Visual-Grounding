from pycocotools.coco import COCO

# Load COCO-formatted dataset with referring expressions
annotation_file = 'merged_coco_annotations.json'
coco = COCO(annotation_file)

# Get all images
image_ids = coco.getImgIds()

# print(image_ids)
# print("number of images", len(image_ids))
# Iterate through images
# total_size = 0
# for img_id in image_ids:
#     img_info = coco.loadImgs(img_id)[0]
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     annotations = coco.loadAnns(ann_ids)
#     total_size += len(annotations)
# print(total_size)
    # for ann in annotations:
    #     bbox = ann['bbox']  # [x, y, width, height]
    #     referring_expression = ann['referring']  # Text query

# Get all category IDs
# category_ids = coco.getCatIds()

# print("All category IDs in the dataset:", category_ids)

img_id = 8
ann_ids = coco.getAnnIds(imgIds=img_id)
annotations = coco.loadAnns(ann_ids)
for ann in annotations:
    print("category_id", ann['category_id'])
    print("bbox", ann['bbox'])


# example predicitons
# [{'image_id': 4, 'category_id': 3, 'bbox': [607, 219, 75, 221], 'score': 0.8395789861679077}, 
# {'image_id': 4, 'category_id': 4, 'bbox': [840, 228, 105, 277], 'score': 0.7893781661987305}, 
# {'image_id': 4, 'category_id': 4, 'bbox': [504, 222, 72, 208], 'score': 0.7663187980651855}, 
# {'image_id': 4, 'category_id': 5, 'bbox': [400, 243, 92, 226], 'score': 0.7300754189491272}, 
# {'image_id': 4, 'category_id': 8, 'bbox': [738, 225, 78, 222], 'score': 0.643081784248352}]

# image id:  8
#[{'image_id': 8, 'category_id': 3, 'bbox': [682, 366, 187, 115], 'score': 0.8589004874229431}, 
# {'image_id': 8, 'category_id': 9, 'bbox': [459, 359, 199, 61], 'score': 0.794265627861023}, 
# {'image_id': 8, 'category_id': 15, 'bbox': [657, 305, 150, 53], 'score': 0.6668002009391785}]