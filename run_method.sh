# extraction models
ems=(
    "gpt-4o-2024-08-06"
    "gpt-4o-mini-2024-07-18"
    "claude-3-5-haiku-20241022"
    "claude-3-7-sonnet-20250219"
    "internvl2_5-1b"
    "internvl2_5-2b"
    "internvl2_5-4b"
    "internvl2_5-8b"
    "qwen2.5-vl-3b"
    "qwen2.5-vl-7b"
    "qwen-vl"
    "qwen-vl-chat"
)

for extraction_model in "${ems[@]}"; do
    # bbox prediction
    CUDA_VISIBLE_DEVICES=2 python predict_bbox.py -e ${extraction_model}

    # bbox matching
    python match_results.py -e ${extraction_model} -m gpt-4o-mini-2024-07-18
    python match_results.py -e ${extraction_model} -m gpt-4o-2024-08-06
    python match_results.py -e ${extraction_model} -m claude-3-5-haiku-latest
    python match_results.py -e ${extraction_model} -m claude-3.7-sonnet-latest

    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m qwen2.5-7b-instruct
    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m qwen2.5-14b-instruct
    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m qwen2.5-32b-instruct
    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m qwen2.5-72b-instruct
    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m llama-3.1-8b-instruct
    CUDA_VISIBLE_DEVICES=2,3,4,5 python match_results.py -e ${extraction_model} -m llama-3.1-70b-instruct

done

