import json
import cv2
import matplotlib.pyplot as plt
import os

# Load the predictions
with open("all_grounding_dino_results.json", "r") as f:
    results = json.load(f)

# Load COCO dataset image mapping
annotation_file = "merged_coco_annotations.json"

with open(annotation_file, "r") as f:
    dataset = json.load(f)

# Map image IDs to file paths
image_mapping = {img["id"]: img["file_name"] for img in dataset["images"]}
image_folder = "/metadisk/label-studio/scenes" 

# image_path = os.path.join(image_folder, image_mapping.get(image_id))

def visualize_prediction(result):
    """
    Visualize the Ground Truth (GT) and Predicted bounding boxes on an image.
    
    :param result: Dictionary containing 'image_id', 'gt_bbox', 'pred_bbox', 'referring', and 'iou'.
    """
    image_id = result["image_id"]
    image_path = image_path = os.path.join(image_folder, image_mapping.get(image_id))

    if image_path is None:
        print(f"Image ID {image_id} not found in dataset!")
        return

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Extract bounding box coordinates
    x, y, w, h = result["gt_bbox"]
    gt_bbox = [x, y, x + w, y + h]  # Convert COCO format to [x_min, y_min, x_max, y_max]

    pred_bbox = result.get("pred_bbox")  # Use .get() to avoid KeyError

    # Check if pred_bbox is None or invalid
    if pred_bbox is None or not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
        print(f"Skipping prediction for image {image_id}, invalid pred_bbox: {pred_bbox}")
        pred_bbox = None

    # Draw Ground Truth bounding box (Green)
    cv2.rectangle(image, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)

    # Draw Predicted bounding box (Red) only if valid
    if pred_bbox is not None:
        try:
            pred_bbox = list(map(int, pred_bbox))  # Ensure values are integers
            cv2.rectangle(image, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)
        except Exception as e:
            print(f"Error drawing predicted box for image {image_id}: {e}")

    # Add text labels
    cv2.putText(image, f"GT: {result['referring']}", (gt_bbox[0], gt_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if pred_bbox is not None:
        cv2.putText(image, f"Pred: IoU {result['iou']:.2f}", (pred_bbox[0], pred_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Referring: {result['referring']}")
    plt.show()

# Select an example (change index for different images)
# example_index = 25  # Change to visualize different samples
# visualize_prediction(results[example_index])

# print(len(results))

def visualize_all_predictions(image_id):
    """
    Visualize all referring expressions and bounding boxes for a single image.
    
    :param image_id: The ID of the image to visualize.
    """
    image_path = os.path.join(image_folder, image_mapping.get(image_id))

    if image_path is None:
        print(f"Image ID {image_id} not found in dataset!")
        return

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Filter results for the given image_id
    image_results = [res for res in results if res["image_id"] == image_id]

    for res in image_results:
        referring_expr = res["referring"]
        x, y, w, h = res["gt_bbox"]
        gt_bbox = [x, y, x + w, y + h]  # Convert COCO format to [x_min, y_min, x_max, y_max]
        pred_bbox = res.get("pred_bbox")

        # Draw Ground Truth bounding box (Green)
        cv2.rectangle(image, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, referring_expr, (gt_bbox[0], gt_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Predicted bounding box (Red) if exists
        if pred_bbox and isinstance(pred_bbox, list) and len(pred_bbox) == 4:
            try:
                pred_bbox = list(map(int, pred_bbox))  # Ensure values are integers
                cv2.rectangle(image, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)
                cv2.putText(image, f"IoU: {res['iou']:.2f}", (pred_bbox[0], pred_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error drawing predicted box for image {image_id}: {e}")

    # Display the image
    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Referring Expressions for Image ID {image_id}")
    plt.show()

#  Choose an image ID to visualize (change as needed)
image_id_to_visualize = 70  # Modify this to visualize another image
visualize_all_predictions(image_id_to_visualize)
