import json
import argparse

from tqdm import tqdm

from eval_utils.match import expression_match, expression_one_match

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--extraction-model', help='extraction model: gpt-4o, gpt-4o-mini, claude-3-haiku')
parser.add_argument('-m', '--match-model', help='match model: gpt-4o, gpt-4o-mini, claude-3-haiku')
args = parser.parse_args()

def bbox_iou_xywh(box1, box2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes in xywh format.

    Args:
        box1 (list or tuple): [x, y, width, height] of first bounding box.
        box2 (list or tuple): [x, y, width, height] of second bounding box.

    Returns:
        float: IoU value (0 to 1).
    """
    # Convert from xywh to xyxy (x_min, y_min, x_max, y_max)
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Compute intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection area
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Compute areas of both bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Compute IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def match_results():
    with open(f'results/NIDS-Net_predictions_{args.extraction_model}.json', 'r') as f:
        match_preps = json.load(f)

    total_size = 0
    eval_results = []

    for match_prep in tqdm(match_preps):
        image_id = match_prep['image_id']
        predictions = match_prep['predictions']
        gt_anns = match_prep['gt_anns']

        # Extract referring expressions from ground truth
        gt_refs = [ann.get('referring', "") for ann in gt_anns]
        gt_bboxes = [ann.get('bbox', "") for ann in gt_anns]

        total_size += len(gt_anns)

        # match predictions with gt_refs
        for ridx, cur_ref in enumerate(gt_refs):
            one_match = expression_one_match(predictions, cur_ref, args.match_model)
            pred_bbox_id = one_match.get('item_id', None)
            if (pred_bbox_id is None or (not isinstance(pred_bbox_id, int)) or
                    pred_bbox_id < 0 or pred_bbox_id >= len(predictions)):
                pred_bbox = [0, 0, 0, 0]
                iou = 0
            else:
                pred_bbox = predictions[pred_bbox_id]['bbox']
                iou = bbox_iou_xywh(gt_bboxes[ridx], pred_bbox)
            eval_results.append({
                "image_id": image_id,
                "referring": gt_refs[ridx],
                "gt_bbox": gt_bboxes[ridx],
                "pred_bbox": pred_bbox,
                "iou": float(iou)
            })

    # save predictions
    with open(f"results/our_results_{args.extraction_model}_{args.match_model}_0408_test_all_one_for_one.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    # Compute Accuracy
    correct_predictions = sum(1 for result in eval_results if result["iou"] > 0.5)
    total_pred = len(eval_results)
    accuracy = correct_predictions / total_size if total_size > 0 else 0
    # Print Accuracy
    print(f"Total number of predictions: {total_pred}")
    print(f"Total samples: {total_size}")
    print(f"Accuracy (IoU > 0.5): {accuracy * 100:.2f}%")
    result_info = {
        'extraction_model': args.extraction_model,
        'match_model': args.match_model,
        'accuracy (IoU > 0.5)': accuracy,
    }
    with open('results/overall_results.jsonl', 'a') as f:
        f.write(json.dumps(result_info)+'\n')

if '__main__' == __name__:
    match_results()

