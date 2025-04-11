import json
import os
import json
import cv2
from ros.nids_net import NIDS
import torch
from PIL import Image
from pycocotools.coco import COCO
import json
from eval_utils.few_shot_VLM import ObjectDescriptionLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--extraction-model', help='extraction model: gpt-4o, gpt-4o-mini, claude-3-haiku')
args = parser.parse_args()

# load NIDS-Net
labels = ['background','001_a_and_w_root_beer_soda_pop_bottle', '002_coca-cola_soda_diet_pop_bottle', '003_coca-cola_soda_original_pop_bottle', '004_coca-cola_soda_zero_pop_bottle', '005_dr_pepper_soda_pop_bottle', '006_fanta_orange_fruit_soda_pop_bottle', '007_powerade_mountain_berry_blast', '008_powerade_zero_purple_grape', '009_samuel_adams_boston_lager_craft_beer', '010_sprite_lemon_lime_soda_pop_bottle', '011_fanta_strawberry_soda_bottle', '012_coca-cola_cherry_soda_pop_bottle', '013_tropicana_cranberry_juice', '014_monster_energy_mega_can', '015_barqs_root_beer_soda_bottle', '016_fairlife_reduced_fat_milk', '017_vita_coco_the_original_coconut_water', '018_chobani_pumpkin_spice_oat_coffee_creamer', '019_dunkin_original_iced_coffee', '020_pure_life_purified_water', '021_comet_no_scent_soft_cleaner_with_bleach', '022_head_and_shoulders_shampoo_and_conditioner', '023_dove_deep_body_wash', '024_seventh_generation_toilet_bowl_cleaner', '025_lysol_power_toilet_bowl_cleaner_gel', '026_woolite_extra_delicates_laundry_detergent', '027_raw_sugar_mens_body_wash', '028_ty-d-bol_rust_stain_remover', '029_palmers_cocoa_butter_formula_massage_lotion', '030_body_moisturizer_by_cetaphil', '031_hi-chew_berry_mix_peg_bag', '032_hi-chew_superfruit_mix_peg_bag', '033_popin_cookin_tanoshii_hamburger_diy_candy', '034_popin_cookin_tanoshii_donuts_diy_candy', '035_pocky_strawberry_cream_sticks', '036_pocky_banana_cream_sticks', '037_pocky_chocolate_cream_sticks', '038_pocky_crunchy_strawberry_cream_sticks', '039_pocky_cookies_and_cream_sticks', '040_pocky_almond_crush_chocolate_cream_sticks', '041_calpico_melon_drink', '042_calpico_mango_drink', '043_calpico_strawberry_drink', '044_meiji_choco_macadamia', '045_meiji_choco_almond', '046_horizon_organic_whole_milk', '047_pillsbury_chocolate_fudge', '048_vita_coco_coconut_milk', '049_tazo_green_tea_matcha_latte_concentrate', '050_golden_curry_japanese_curry_mix', '051_equate_baby_powder', '052_native_body_wash', '053_arm_and_hammer_baking_soda', '054_kodiak_cakes_power_cakes_flapjack_quick_mix', '055_bosco_chocolate_syrup', '056_hairitage_body_lotion', '057_lipton_kosher_soup_recipe_vegetable', '058_ritz_original_crackers', '059_quaker_instant_oatmeal', '060_nesquik_chocolate_powder', '061_sweet_baby_rays_original_barbecue_sauce', '062_hain_pure_foods_sea_salt', '063_lays_stax_potato_crisps', '064_soothing_body_wash', '065_bacon_grease', '066_snuggle_fabric_softener_sheets', '067_nesquik_strawberry_syrup', '068_crest_scope_liquid_gel_toothpaste', '069_native_deodorant', '070_shout_advanced_action_gel', '071_coco_real_cream_of_coconut', '072_butter_original_spray', '073_mosquito_repellent_spritz', '074_heinz_mayomust_sauce', '075_method_men_gel_liquid_body_wash', '076_instant_cappuccino_mix', '077_osem_natural_soup_mix', '078_super_coffee_vanilla_creamer', '079_reynolds_cut-rite_wax_paper', '080_great_value_whole_wheat_spaghetti', '081_hersheys_cocoa_powder', '082_lifeway_organic_whole_milk_peach_kefir', '083_body_proud_body_wash_cleanser', '084_elmers_school_glue', '085_nellie_and_joes_key_west_lime_juice', '086_kens_steak_house_lite_honey_mustard_salad_dressing', '087_great_value_strawberry_syrup', '088_log_cabin_all_natural_table_syrup', '089_hidden_valley_original_ranch', '090_off_deep_woods_dry_mosquito_spray', '091_butter_tub', '092_organic_coconut_oil_and_ghee', '093_kids_bubble_bath_and_body_wash', '094_oats_mixed_berry_oatmeal', '095_hairitage_body_scrub', '096_honey_hot_barbecue_sauce', '097_skin_moisturizer', '098_arm_and_hammmer_liquid_laundry', '099_drano_max_gel_clog_remover', '100_table_tennis_racket']
image_folder = "scenes" 
#annotation_file = #'/metadisk/label-studio/referring_coco_annotation/project_02_coco.json' # 
annotation_file = "merged_coco_annotations.json"

# load object infos
# Example Usage
# Replace with your actual JSONL file path
jsonl_path = f"results/descriptions_{args.extraction_model}.jsonl"
if not os.path.exists(jsonl_path):
    raise Error
loader = ObjectDescriptionLoader(jsonl_path)


# load the model
adapter_descriptors_path = "adapted_obj_feats/refer_weight_1004_temp_0.05_epoch_640_lr_0.001_bs_1024_vec_reduction_4.json"
with open(os.path.join(adapter_descriptors_path), 'r') as f:
    feat_dict = json.load(f)

object_features = torch.Tensor(feat_dict['features']).cuda()
object_features = object_features.view(-1, 14, 1024)
weight_adapter_path = "adapter_weights/refer_weight_1004_temp_0.05_epoch_640_lr_0.001_bs_1024_vec_reduction_4_weights.pth"
model = NIDS(object_features, use_adapter=True, adapter_path=weight_adapter_path, gdino_threshold=0.4, class_labels=labels, dinov2_encoder='dinov2_vitl14_reg')

def load_coco_annotations(gt_json_path):
    """Load COCO-formatted dataset using pycocotools."""
    coco = COCO(gt_json_path)
    return coco

def predict_bbox(gt_json_path, detection_model):
    """Iterate over each image, run the detection model, and get ground truth referring expressions."""
    coco = load_coco_annotations(gt_json_path)
    
    image_ids = coco.getImgIds()  # Get all image IDs
#    eval_results = []
#    total_size = 0
    
    match_preps = []
#    for image_id in tqdm(image_ids[:10]):
    for image_id in tqdm(image_ids):
        img_info = coco.loadImgs(image_id)[0]
        image_path = img_info['file_name']  # Modify if needed
        query_img_path = os.path.join(image_folder, image_path)
        #print("query image: ", query_img_path)
        # query_img_path = image_path
        # img_pil = Image.open(query_img_path)
        # img_pil.show()
        img = cv2.imread(query_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        # Run detection model (Assuming it returns bounding boxes and labels)
        # results, mask = detection_model.step(img, visualize=True)
        results, mask = detection_model.step(img, visualize=False)
        predictions = []
        for res in results:
            pred = {}
            init_bbox = res['bbox'].tolist() # xyxy
            bbox = [init_bbox[0], init_bbox[1], init_bbox[2]-init_bbox[0], init_bbox[3]-init_bbox[1]]
            bbox = [int(i) for i in bbox]
            pred['label'] = res['label']
            pred['bbox'] = bbox #res['bbox']
            pred['category_id'] = res['category_id']
            pred['object_info'] = loader.get_description(pred['label'])
            predictions.append(pred)

        # Get ground truth annotations for this image
        ann_ids = coco.getAnnIds(imgIds=image_id)
        gt_anns = coco.loadAnns(ann_ids)

#        total_size += len(gt_anns)

        match_prep = {
            'image_id': image_id,
            'predictions': predictions,
            'gt_anns': gt_anns,
        }
        match_preps.append(match_prep)

    with open(f'results/NIDS-Net_predictions_{args.extraction_model}.json', 'w') as f:
        json.dump(match_preps, f)

#
#    del coco
#
#    for match_prep in tqdm(match_preps):
#        print(json.dumps(match_prep, indent=2))
#        image_id = match_prep['image_id']
#        predictions = match_prep['predictions']
#        gt_refs = match_prep['gt_refs']
#        gt_bboxes = match_prep['gt_bboxes']
#
#        # match predictions with gt_refs
#        for ridx, cur_ref in enumerate(gt_refs):
#            one_match = expression_one_match(predictions, cur_ref, args.match_model)
#            pred_bbox_id = one_match['item_id']
#            if pred_bbox_id < 0 or pred_bbox_id >= len(predictions):
#                iou = 0
#            else:
#                pred_bbox = predictions[pred_bbox_id]['bbox']
#                iou = bbox_iou_xywh(gt_bboxes[ridx], pred_bbox)
#            eval_results.append({
#                "image_id": image_id,
#                "referring": gt_refs[ridx],
#                "gt_bbox": gt_bboxes[ridx],
#                "pred_bbox": predictions[pred_bbox_id]['bbox'],
#                "iou": float(iou)
#            })

#        final_results = expression_match(predictions, gt_refs, args.match_model)
#        gt_refs_set = set(gt_refs)
#        print("final_results: ", final_results)
#        # print("start matching")
#        for match in final_results:
#            gt_bbox_id = match['inquiry_id']
#            pred_bbox_id = match['item_id']
#            # print("gt_bbox_id: ", gt_bbox_id)
#            # print("pred_bbox_id: ", pred_bbox_id)
#            cur_ref = gt_refs[gt_bbox_id]
#            if cur_ref in gt_refs_set:
#                gt_refs_set.remove(gt_refs[gt_bbox_id])
#            # else:
#            #     continue
#            if (gt_bbox_id < 0 or pred_bbox_id < 0 or
#            gt_bbox_id >= len(gt_bboxes) or pred_bbox_id >= len(predictions)):
#                iou = 0
#            else:
#                pred_bbox = predictions[pred_bbox_id]['bbox']
#
#                iou = bbox_iou_xywh(gt_bboxes[gt_bbox_id], pred_bbox)
#            eval_results.append({
#                "image_id": image_id,
#                "referring": gt_refs[gt_bbox_id],
#                "gt_bbox": gt_bboxes[gt_bbox_id],
#                "pred_bbox": predictions[pred_bbox_id]['bbox'],
#                "iou": float(iou)
#            })
#
#        # if there are unmatched gt_refs, add them to the results
#        for gt_ref in gt_refs_set:
#            eval_results.append({
#                "image_id": image_id,
#                "referring": gt_ref,
#                "gt_bbox": gt_bboxes[gt_refs.index(gt_ref)],
#                "pred_bbox": [],
#                "iou": 0.0
#            })

#    # save predictions
##    with open("VLM4o_our_results_4o_0328_test.json", "w") as f:
##    with open("our_results_4o_0327_test_all_one_for_one.json", "w") as f:
#    with open(f"our_results_{args.match_model}_0407_test_all_one_for_one.json", "w") as f:
#        json.dump(eval_results, f, indent=4)
#    # Compute Accuracy
#    correct_predictions = sum(1 for result in eval_results if result["iou"] > 0.5)
#    total_pred = len(eval_results)
#    accuracy = correct_predictions / total_size if total_size > 0 else 0
#    # Print Accuracy
#    print(f"Total number of predictions: {total_pred}")
#    print(f"Total samples: {total_size}")
#    print(f"Accuracy (IoU > 0.5): {accuracy * 100:.2f}%")

if '__main__' == __name__:
    # Example usage
    gt_json_path = "merged_coco_annotations.json"
    predict_bbox(gt_json_path, model)

