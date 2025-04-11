#CUDA_VISIBLE_DEVICES=4,5 python eval_utils/extract_info.py -e internvl2_5-1b
#CUDA_VISIBLE_DEVICES=4,5 python eval_utils/extract_info.py -e internvl2_5-2b
#CUDA_VISIBLE_DEVICES=4,5 python eval_utils/extract_info.py -e internvl2_5-4b
#CUDA_VISIBLE_DEVICES=4,5 python eval_utils/extract_info.py -e internvl2_5-8b
#CUDA_VISIBLE_DEVICES=2,3,4,5 python eval_utils/extract_info.py -e qwen2.5-vl-3b
#CUDA_VISIBLE_DEVICES=2,3,4,5 python eval_utils/extract_info.py -e qwen2.5-vl-7b
CUDA_VISIBLE_DEVICES=2,3,4,5 python eval_utils/extract_info.py -e qwen-vl
CUDA_VISIBLE_DEVICES=2,3,4,5 python eval_utils/extract_info.py -e qwen-vl-chat

#python eval_utils/extract_info.py -e gpt-4o-mini-2024-07-18
#python eval_utils/extract_info.py -e gpt-4o-2024-08-06
#python eval_utils/extract_info.py -e claude-3-5-haiku-20241022
#python eval_utils/extract_info.py -e claude-3-7-sonnet-20250219

