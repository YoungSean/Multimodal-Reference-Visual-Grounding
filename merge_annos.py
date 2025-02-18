import json
import glob
import os

# Define the directory containing the scene JSON files
json_dir = "./referring_coco_annotation"
output_file = "merged_coco_annotations.json"

# Get list of all JSON files in the directory
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

# Initialize merged dataset structure
merged_coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

image_id_offset = 0
annotation_id_offset = 0
category_id_map = {}  # Maps old category_id to new category_id
category_name_map = {}  # Maps category names to category_id

for json_file in json_files:
    with open(json_file, "r") as f:
        coco_data = json.load(f)
    
    # Merge categories, ensuring unique category names
    for cat in coco_data["categories"]:
        cat_name = cat["name"]
        if cat_name not in category_name_map:
            new_category_id = len(category_name_map) + 1
            category_name_map[cat_name] = new_category_id
            category_id_map[cat["id"]] = new_category_id
            cat["id"] = new_category_id
            merged_coco["categories"].append(cat)
        else:
            category_id_map[cat["id"]] = category_name_map[cat_name]
    
    # Update image IDs
    image_id_map = {}
    for img in coco_data["images"]:
        new_image_id = img["id"] + image_id_offset
        image_id_map[img["id"]] = new_image_id
        img["id"] = new_image_id
        merged_coco["images"].append(img)

    # Update annotation IDs and category references
    for ann in coco_data["annotations"]:
        ann["id"] += annotation_id_offset
        ann["image_id"] = image_id_map[ann["image_id"]]
        ann["category_id"] = category_id_map[ann["category_id"]]
        merged_coco["annotations"].append(ann)

    # Update offsets to avoid ID conflicts
    image_id_offset = max(img["id"] for img in merged_coco["images"]) + 1
    annotation_id_offset = max(ann["id"] for ann in merged_coco["annotations"]) + 1

# Save merged dataset
with open(output_file, "w") as f:
    json.dump(merged_coco, f, indent=4)

print(f"Merged dataset saved to {output_file}")
