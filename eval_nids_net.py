import json
import os
import json
import cv2
from ros.nids_net import NIDS
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm

# load NIDS-Net
labels = ['background','001_a_and_w_root_beer_soda_pop_bottle', '002_coca-cola_soda_diet_pop_bottle', '003_coca-cola_soda_original_pop_bottle', '004_coca-cola_soda_zero_pop_bottle', '005_dr_pepper_soda_pop_bottle', '006_fanta_orange_fruit_soda_pop_bottle', '007_powerade_mountain_berry_blast', '008_powerade_zero_purple_grape', '009_samuel_adams_boston_lager_craft_beer', '010_sprite_lemon_lime_soda_pop_bottle', '011_fanta_strawberry_soda_bottle', '012_coca-cola_cherry_soda_pop_bottle', '013_tropicana_cranberry_juice', '014_monster_energy_mega_can', '015_barqs_root_beer_soda_bottle', '016_fairlife_reduced_fat_milk', '017_vita_coco_the_original_coconut_water', '018_chobani_pumpkin_spice_oat_coffee_creamer', '019_dunkin_original_iced_coffee', '020_pure_life_purified_water', '021_comet_no_scent_soft_cleaner_with_bleach', '022_head_and_shoulders_shampoo_and_conditioner', '023_dove_deep_body_wash', '024_seventh_generation_toilet_bowl_cleaner', '025_lysol_power_toilet_bowl_cleaner_gel', '026_woolite_extra_delicates_laundry_detergent', '027_raw_sugar_mens_body_wash', '028_ty-d-bol_rust_stain_remover', '029_palmers_cocoa_butter_formula_massage_lotion', '030_body_moisturizer_by_cetaphil', '031_hi-chew_berry_mix_peg_bag', '032_hi-chew_superfruit_mix_peg_bag', '033_popin_cookin_tanoshii_hamburger_diy_candy', '034_popin_cookin_tanoshii_donuts_diy_candy', '035_pocky_strawberry_cream_sticks', '036_pocky_banana_cream_sticks', '037_pocky_chocolate_cream_sticks', '038_pocky_crunchy_strawberry_cream_sticks', '039_pocky_cookies_and_cream_sticks', '040_pocky_almond_crush_chocolate_cream_sticks', '041_calpico_melon_drink', '042_calpico_mango_drink', '043_calpico_strawberry_drink', '044_meiji_choco_macadamia', '045_meiji_choco_almond', '046_horizon_organic_whole_milk', '047_pillsbury_chocolate_fudge', '048_vita_coco_coconut_milk', '049_tazo_green_tea_matcha_latte_concentrate', '050_golden_curry_japanese_curry_mix', '051_equate_baby_powder', '052_native_body_wash', '053_arm_and_hammer_baking_soda', '054_kodiak_cakes_power_cakes_flapjack_quick_mix', '055_bosco_chocolate_syrup', '056_hairitage_body_lotion', '057_lipton_kosher_soup_recipe_vegetable', '058_ritz_original_crackers', '059_quaker_instant_oatmeal', '060_nesquik_chocolate_powder', '061_sweet_baby_rays_original_barbecue_sauce', '062_hain_pure_foods_sea_salt', '063_lays_stax_potato_crisps', '064_soothing_body_wash', '065_bacon_grease', '066_snuggle_fabric_softener_sheets', '067_nesquik_strawberry_syrup', '068_crest_scope_liquid_gel_toothpaste', '069_native_deodorant', '070_shout_advanced_action_gel', '071_coco_real_cream_of_coconut', '072_butter_original_spray', '073_mosquito_repellent_spritz', '074_heinz_mayomust_sauce', '075_method_men_gel_liquid_body_wash', '076_instant_cappuccino_mix', '077_osem_natural_soup_mix', '078_super_coffee_vanilla_creamer', '079_reynolds_cut-rite_wax_paper', '080_great_value_whole_wheat_spaghetti', '081_hersheys_cocoa_powder', '082_lifeway_organic_whole_milk_peach_kefir', '083_body_proud_body_wash_cleanser', '084_elmers_school_glue', '085_nellie_and_joes_key_west_lime_juice', '086_kens_steak_house_lite_honey_mustard_salad_dressing', '087_great_value_strawberry_syrup', '088_log_cabin_all_natural_table_syrup', '089_hidden_valley_original_ranch', '090_off_deep_woods_dry_mosquito_spray', '091_butter_tub', '092_organic_coconut_oil_and_ghee', '093_kids_bubble_bath_and_body_wash', '094_oats_mixed_berry_oatmeal', '095_hairitage_body_scrub', '096_honey_hot_barbecue_sauce', '097_skin_moisturizer', '098_arm_and_hammmer_liquid_laundry', '099_drano_max_gel_clog_remover', '100_table_tennis_racket']
image_folder = "/metadisk/label-studio/scenes" 
annotation_file = "merged_coco_annotations.json"


# load the model
epoch = 320
# model_weight = f"refer_weight_1004_temp_0.05_epoch_{epoch}_lr_0.001_bs_1024_vec_reduction_4"
model_weight = f"refer_weight_beta_10.0_031325_temp_0.05_epoch_320_lr_0.001_bs_1400_vec_reduction_4"
adapter_descriptors_path = f"adapted_obj_feats/{model_weight}.json"
with open(os.path.join(adapter_descriptors_path), 'r') as f:
    feat_dict = json.load(f)

object_features = torch.Tensor(feat_dict['features']).cuda()
object_features = object_features.view(-1, 14, 1024)
weight_adapter_path = f"adapter_weights/{model_weight}_weights.pth"

model = NIDS(object_features, use_adapter=False, adapter_path=weight_adapter_path, gdino_threshold=0.4, class_labels=labels, dinov2_encoder='dinov2_vitl14_reg')


def process_images_with_model(gt_json_path, detection_model, image_folder=image_folder):
    """Iterate over each image, run the detection model, and get ground truth referring expressions."""
    coco = COCO(gt_json_path)
    
    image_ids = coco.getImgIds()  # Get all image IDs
    predictions = []
    for image_id in tqdm(image_ids):
        #print("image id: ", image_id)
        img_info = coco.loadImgs(image_id)[0]
        image_path = img_info['file_name']  # Modify if needed
        query_img_path = os.path.join(image_folder, image_path)
        #print("query image: ", query_img_path)
        # query_img_path = image_path
        # img_pil = Image.open(query_img_path)
        # img_pil.show()
        img = cv2.imread(query_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run detection model (Assuming it returns bounding boxes)
        results, mask = detection_model.step(img, visualize=False)
        
        for res in results:
            init_bbox = res['bbox'].tolist() # xyxy
            bbox = [init_bbox[0], init_bbox[1], init_bbox[2]-init_bbox[0], init_bbox[3]-init_bbox[1]] #xywh
            bbox = [int(i) for i in bbox]
            predictions.append({
                "image_id": image_id,
                "category_id": int(res['category_id']),
                "bbox": bbox,
                "score": float(res['score'])
            })
    return predictions

def evaluate_coco(gt_path, pred_path):
    """
    Evaluate a COCO-format object detection model.

    Args:
        gt_path (str): Path to the ground truth annotations file.
        pred_path (str): Path to the predicted annotations file.

    Returns:
        dict: Evaluation results including mAP and AR.
    """
    # Load ground truth annotations
    coco_gt = COCO(gt_path)

    # Load predicted annotations
    coco_dt = coco_gt.loadRes(pred_path)

    # Initialize COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract results
    results = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8]
    }

    return results


predictions = process_images_with_model(annotation_file, model)
pred_annotations_path = os.path.join("results", f"basic_nids_net.json")
with open(pred_annotations_path, 'w') as f:
    json.dump(predictions, f, indent=4)

# Evaluate the model
results = evaluate_coco(annotation_file, pred_annotations_path)
print("Evaluation Results:", results)






# Get ground truth annotations for this image
# image_id = 4
# coco_gt = COCO(annotation_file)
# img_info = coco_gt.loadImgs(image_id)[0]
# ann_ids = coco_gt.getAnnIds(imgIds=image_id)
# annotations = coco_gt.loadAnns(ann_ids)

# # Create a mini COCO-style JSON with only this image's annotations
# gt_subset = {
#     "images": [img_info],
#     "annotations": annotations,
#     "categories": coco_gt.loadCats(coco_gt.getCatIds())
# }
# print("gt_subset", gt_subset)

# gt_subset_path = "ground_truth_single_8.json"
# with open(gt_subset_path, "w") as f:
#     json.dump(gt_subset, f)

# # Load model and run inference
# predictions = process_images_with_model(annotation_file, model)
# print("predictions: ", predictions)
# # Save predictions to JSON
# pred_subset_path = "predictions_single_8.json"
# with open(pred_subset_path, "w") as f:
#     json.dump(predictions, f)

# # Evaluate using COCO API
# coco_gt_subset = COCO(gt_subset_path)
# coco_dt = coco_gt_subset.loadRes(pred_subset_path)
# coco_eval = COCOeval(coco_gt_subset, coco_dt, iouType="bbox")

# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

# # Extract results
# results = {
#     "mAP@0.5:0.95": coco_eval.stats[0],
#     "mAP@0.5": coco_eval.stats[1],
#     "mAP@0.75": coco_eval.stats[2],
#     "AR@1": coco_eval.stats[6],
#     "AR@10": coco_eval.stats[7],
#     "AR@100": coco_eval.stats[8]
# }
# print(results)