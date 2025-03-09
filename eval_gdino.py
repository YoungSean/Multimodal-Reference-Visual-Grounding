
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import json
import os

gdino = GroundingDINOObjectPredictor(threshold=0.2)
#SAM = SegmentAnythingPredictor()


image_folder = "/metadisk/label-studio/scenes" 
#annotation_file = #'/metadisk/label-studio/referring_coco_annotation/project_02_coco.json' # 
annotation_file = "merged_coco_annotations.json"
with open(annotation_file, "r") as f:
    dataset = json.load(f)

images = {img["id"]: img["file_name"] for img in dataset["images"]}
annotations = dataset["annotations"]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# save results
results = []

for ann in annotations:
    image_id = ann["image_id"]
    image_path = os.path.join(image_folder, images[image_id])
    referring_expr = ann["referring"]
    gt_bbox = ann["bbox"]  # [x, y, w, h]


    image_pil = PILImg.open(image_path).convert("RGB")

    # run Grounding DINO
    boxes, phrases, gdino_conf = gdino.predict(image_pil, referring_expr)
    w, h = image_pil.size # Get image width and height 
    # Scale bounding boxes to match the original image size
    #image_pil_bboxes = gdino.bbox_to_scaled_xyxy(boxes, w, h)
    # Ensure there is at least one bounding box predicted
    if len(boxes) > 0:
        # Get the index of the highest confidence bounding box
        best_idx = gdino_conf.argmax().item()  

        # Select the best bounding box and corresponding confidence
        best_box = boxes[best_idx]  # (x_min, y_min, x_max, y_max)
        best_phrase = phrases[best_idx]
        best_conf = gdino_conf[best_idx]

        # Scale bounding box to match the original image size
        best_box_scaled = gdino.bbox_to_scaled_xyxy(best_box.unsqueeze(0), w, h)[0]  # Extract from tensor

        #print(f"Selected Bounding Box: {best_box_scaled} | Confidence: {best_conf:.4f}")
    else:
        best_box_scaled = None
        print("No bounding box detected.")


    best_iou = 0
    best_box = None
    #print(len(best_box_scaled))
    #for box in best_box_scaled:
    iou = compute_iou(best_box_scaled, gt_bbox)
    if iou > best_iou:
        best_iou = float(iou)
        best_box = best_box_scaled.tolist()

    results.append({
        "image_id": image_id,
        "referring": referring_expr,
        "gt_bbox": gt_bbox,
        "pred_bbox": best_box,
        "iou": float(best_iou)
    })

# save predictions
with open("all_best_conf_grounding_dino_results_test10.json", "w") as f:
    json.dump(results, f, indent=4)

# Compute Accuracy
correct_predictions = sum(1 for result in results if result["iou"] > 0.5)
total_samples = len(results)
accuracy = correct_predictions / total_samples if total_samples > 0 else 0

# Print Accuracy
print(f"Grounding DINO Accuracy (IoU > 0.5): {accuracy * 100:.2f}%")



