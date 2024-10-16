import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from scipy.optimize import linear_sum_assignment
from src.model.utils import Detections
import cv2
import torch
from PIL import Image
import glob
import os
from pathlib import Path

gdino_threshold = 0.4
sam_model = "vit_h"

gdino = GroundingDINOObjectPredictor(threshold=gdino_threshold)
SAM = SegmentAnythingPredictor(vit_model=sam_model)


def get_template_mask_per_image(image_path, prompt="bottle"):
    """
    Get template features from the template image of one object. This function can be used for one-shot detection.
    Parameters
    ----------
    template_image_pil: RGB PIL image
    mask: a torch tensor, [H,W] with unique values for each object
    -------
    """
    image_pil = PILImg.open(image_path)
    image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)
    # image_pil.show()
    bboxes, phrases, gdino_conf = gdino.predict(image_pil, prompt)
    w, h = image_pil.size  # Get image width and height
    # Scale bounding boxes to match the original image size
    image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
    image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
    bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)
    bbox_annotated_pil.show()

    assert len(masks) == 1, f"Only one object is allowed in the template image, but {image_path} has {len(masks)} objects."
    mask = masks[0][0]
    # Convert to NumPy array (0 and 255 for binary image mask)
    numpy_array = mask.cpu().numpy().astype(np.uint8) * 255

    # Convert NumPy array to PIL Image
    mask_image = Image.fromarray(numpy_array)

    # Save the mask image
    #mask_img_path = image_path.replace("color", "mask").replace(".jpg", ".png")
    prefix = image_path.split('.')[0]
    mask_img_path = f"{prefix}_mask.png"
    mask_image.save(mask_img_path)


# image_path = "color_043422252387_20240913_163542.jpg"
#
# get_template_mask_per_image(image_path)


# Define the folder path
new_folders = []
folder_path = Path('/metadisk/label-studio/templates').resolve()
# print(os.listdir(folder_path))
folders = [f for f in folder_path.iterdir() if f.is_dir()]

# for folder in folders:
#     folder_name = folder.name
#     folder_name = folder_name.replace('\u200b', '')
#     folder_name = folder_name.replace(' ', '_')
#     # os.rename(folder.name, folder_name)
#     print(folder_name)
#     new_folders.append(folder_name)
#
# print(new_folders)

folders.sort()
print(folders)
# folder_path = folders[6]
# print(folder_path)

wrong_mask = 0
for folder_path in folders:
    # # Get all image paths where filenames start with 'color' and do not contain 'mask'
    # image_paths = [path for path in glob.glob(f"{folder_path}/color*.[pj][pn]g") if 'mask' not in path.lower()]
    # image_paths.sort()
    #
    # # Loop through each image path and open the image
    # for i, image_path in enumerate(image_paths):
    #     print("Processing image", i, " the path is ", image_path)
    #     get_template_mask_per_image(image_path, "bottle") #
    image_paths = glob.glob(f"{folder_path}/*mask.[pj][pn]g")

    if len(image_paths) != 14:
        print(f"Folder {folder_path} has {len(image_paths)} mask images.")
        wrong_mask += 1

print(f"Total number of folders with wrong mask images: {wrong_mask}")

# get_template_mask_per_image()

# folder_path = folders[99]
# image_paths = [path for path in glob.glob(f"{folder_path}/color*.[pj][pn]g") if 'mask' not in path.lower()]
# image_paths.sort()
# #
# # # Loop through each image path and open the image
# for i, image_path in enumerate(image_paths[8:9]):
#     print("Processing image", i, " the path is ", image_path)
#     get_template_mask_per_image(image_path, "bottle") #red object white bottle small item

# get_template_mask_per_image('/metadisk/label-studio/templates/sprite_lemon_lime_soda_pop_bottle/color_046122250168_20240913_164259.jpg', "bottle") #