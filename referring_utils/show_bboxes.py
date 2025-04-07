import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
image_path = "/metadisk/label-studio/scenes/scene_025/color_239222302862_20240927_170411.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the two bounding boxes (x_min, y_min, x_max, y_max)
image_id = 3676
referring = "a red bottle showing its side"
mode = "xyxy" # xyxy or xywh
model = "groundDINO" # internVL 2.5, Qwen 2.5 VL, groundDINO, ours, gt
bbox1 = (651,
            302,
            683,
            400
) #( 533,542,533+94,542+112)   # Example box 1
bbox2 = (481, 449, 583, 526) # Example box 2
gt_bbox= ()

# Create a plot
fig, ax = plt.subplots(1)
ax.imshow(image)

if mode == "xyxy":
    # Create rectangle patches and add them to the plot
    rect1 = patches.Rectangle((bbox1[0], bbox1[1]), bbox1[2]-bbox1[0], bbox1[3]-bbox1[1],
                              linewidth=2, edgecolor='lime', facecolor='none')
else:
    rect1 = patches.Rectangle((bbox1[0], bbox1[1]), bbox1[2], bbox1[3],
                              linewidth=2, edgecolor='lime', facecolor='none')
# rect2 = patches.Rectangle((bbox2[0], bbox2[1]), bbox2[2]-bbox2[0], bbox2[3]-bbox2[1],
#                           linewidth=2, edgecolor='red', facecolor='none')

ax.add_patch(rect1)
# ax.add_patch(rect2)

# Display
plt.axis('off')
plt.tight_layout()
plt.savefig(f"VLM_compare/comparsion_{image_id}_{referring}_{model}.png", dpi=300, bbox_inches='tight', pad_inches=0, format='png')
plt.show()