import json
import os

# Path to the merged COCO JSON file
coco_json_path = "merged_coco_annotations.json"

def validate_coco(coco_json_path):
    try:
        # Load JSON file
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        # Required keys in COCO format
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                print(f"Error: Missing key '{key}' in COCO JSON")
                return False

        # Validate images
        image_ids = set()
        for img in coco_data["images"]:
            if "id" not in img or "file_name" not in img or "width" not in img or "height" not in img:
                print(f"Error: Invalid image entry: {img}")
                return False
            if img["id"] in image_ids:
                print(f"Error: Duplicate image_id {img['id']}")
                return False
            image_ids.add(img["id"])

        # Validate categories
        category_ids = set()
        for cat in coco_data["categories"]:
            if "id" not in cat or "name" not in cat:
                print(f"Error: Invalid category entry: {cat}")
                return False
            if cat["id"] in category_ids:
                print(f"Error: Duplicate category_id {cat['id']}")
                return False
            category_ids.add(cat["id"])

        # Validate annotations
        annotation_ids = set()
        for ann in coco_data["annotations"]:
            if "id" not in ann or "image_id" not in ann or "category_id" not in ann or "bbox" not in ann:
                print(f"Error: Invalid annotation entry: {ann}")
                return False
            if ann["id"] in annotation_ids:
                print(f"Error: Duplicate annotation_id {ann['id']}")
                return False
            if ann["image_id"] not in image_ids:
                print(f"Error: Annotation {ann['id']} references missing image_id {ann['image_id']}")
                return False
            if ann["category_id"] not in category_ids:
                print(f"Error: Annotation {ann['id']} references missing category_id {ann['category_id']}")
                return False
            if len(ann["bbox"]) != 4 or any(x < 0 for x in ann["bbox"]):
                print(f"Error: Invalid bbox format in annotation {ann['id']}")
                return False
            annotation_ids.add(ann["id"])

        print("COCO JSON validation successful! ✅")
        return True

    except json.JSONDecodeError:
        print("Error: Invalid JSON format!")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Run validation
validate_coco(coco_json_path)
