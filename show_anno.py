import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import random


# Load the COCO-style JSON file
file_path = '/metadisk/label-studio/referring_coco_annotation/project_01_coco.json'
with open(file_path, 'r') as file:
    coco_data = json.load(file)

# Function to visualize an image with annotations
def visualize_annotations(image_id, coco_data, image_folder):
    # Find the image info by ID
    image_info = next((img for img in coco_data["images"] if img["id"] == image_id), None)
    if not image_info:
        print(f"Image with ID {image_id} not found.")
        return

    # Load the image
    image_path = f"{image_folder}/{image_info['file_name']}"  # Update `image_folder` with the path to your images
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image file {image_path} not found.")
        return

    # Get annotations for the image
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')

    # Function to generate random colors
    def random_color():
        return [random.random() for _ in range(3)]

    for ann in annotations:
        # Bounding box information
        bbox = ann["bbox"]
        category_id = ann["category_id"]
        referring = ann.get("referring", "")
        x, y, width, height = bbox

        # Draw bounding box
        plt.gca().add_patch(Rectangle((x, y), width, height, edgecolor='red', facecolor='none', linewidth=2))

        # Draw segmentation mask with a unique color
        if "segmentation" in ann and len(ann["segmentation"]) > 0:
            mask_color = random_color()
            for seg in ann["segmentation"]:
                plt.plot(seg[::2], seg[1::2], linestyle='-', linewidth=1.5, color=mask_color, alpha=0.8)

        plt.text(x, y - 5, f"ID: {category_id}, {referring}", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))


    plt.title(f"Image ID: {image_id}")
    plt.show()

# Example usage
image_folder = "/metadisk/label-studio/scenes"  # Replace with the folder where your images are stored
image_id_to_visualize = 4  # Replace with the ID of the image you want to visualize
visualize_annotations(image_id_to_visualize, coco_data, image_folder)
