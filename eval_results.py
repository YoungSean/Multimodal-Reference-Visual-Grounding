import json

import numpy as np

def compute_accuracy_metrics(ious):
    """
    Args:
        ious (list or np.array): IoU values for each sample.

    Returns:
        acc_050: Accuracy at IoU >= 0.5
        acc_075: Accuracy at IoU >= 0.75
        acc_090: Accuracy at IoU >= 0.9
        macc: Mean accuracy from IoU thresholds 0.5 to 0.9 (step 0.05)
    """
    ious = np.array(ious)

    # Acc@0.5
    acc_050 = np.mean(ious >= 0.5)

    # Acc@0.75
    acc_075 = np.mean(ious >= 0.75)

    # Acc@0.9
    acc_090 = np.mean(ious >= 0.9)

    # mAcc: mean of Acc@0.5, 0.55, ..., 0.9
    thresholds = np.arange(0.5, 0.95, 0.05)
    accs = [(ious >= t).mean() for t in thresholds]
    macc = np.mean(accs)

    return acc_050, acc_075, acc_090, macc

###################################################
##### Replace with your actual path if needed #####
###################################################
file_path = "rssults/our_results_gpt-4o-2024-08-06_0328_test_all_one_for_one.json"

# Open and load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Now `data` contains the JSON content as Python dict/list

ious = [result["iou"] for result in data ]  #
print(len(ious))
if len(ious) < 855:  # if the length is less than 855, pad with zeros. we have 855 samples in total
    ious += [0] * (855 - len(ious))
print(len(ious))
acc_050, acc_075, acc_090, macc = compute_accuracy_metrics(ious)
print(f"Acc@0.5: {acc_050:.4f}")
print(f"Acc@0.75: {acc_075:.4f}")
print(f"Acc@0.90: {acc_090:.4f}")
print(f"mAcc: {macc:.4f}")

