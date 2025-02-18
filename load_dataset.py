from pycocotools.coco import COCO

# Load COCO-formatted dataset with referring expressions
annotation_file = 'merged_coco_annotations.json'
coco = COCO(annotation_file)

# Get all images
image_ids = coco.getImgIds()

print(image_ids)
print("number of images", len(image_ids))
# Iterate through images
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        referring_expression = ann['referring']  # Text query

        print(f"Image: {img_info['file_name']}, Expression: {referring_expression}, GT BBox: {bbox}")
    break
