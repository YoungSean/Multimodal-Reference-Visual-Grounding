from pycocotools.coco import COCO

# Load COCO-formatted dataset with referring expressions
annotation_file = 'merged_coco_annotations.json'
coco = COCO(annotation_file)

# Get all images
image_ids = coco.getImgIds()

img_id = 8
ann_ids = coco.getAnnIds(imgIds=img_id)
annotations = coco.loadAnns(ann_ids)
for ann in annotations:
    print("category_id", ann['category_id'])
    print("bbox", ann['bbox'])

