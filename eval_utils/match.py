import json

from pydantic import BaseModel

#def init_matching_model(model_name):
#    if 'gpt-4' == model_name[:6]:
#        from eval_utils import openai_api
#    elif 'claude' == model_name[:6]:
#        from eval_utils import anthropic_api
#    elif 'llama' == model_name[:5] or 'qwen' == model_name[:4]:
#        from eval_utils import vllm_api
from eval_utils import openai_api
from eval_utils import anthropic_api
from eval_utils import vllm_api

class OneMatch(BaseModel):
    item_id: int

class Match(BaseModel):
    inquiry_id: int
    item_id: int

class AllMatches(BaseModel):
    matches: list[Match]

system_prompt = (
'You are an expert in information matching. '
#'Your task is to match items from a given list of descriptions to corresponding inquiries based on relevance. '
'Your task is to match items from a given list of descriptions to the given inquiry based on relevance. '
'Each inquiry only matches one item description and appears once in the final output.\n'
'Each item description includes positional information, where the first value represents the x-axis '
'(horizontal position) and the second value represents the y-axis (vertical position). '
'A higher x-axis value indicates the item is positioned further to the right. '
'A higher y-axis value indicates the item is positioned lower.\n'
'Once you determine the matches, convert them into the specified output format.'
)

group_prompt_template = (
'Items\' Description:\n'
'{descs}\n'
'Inquiries:\n'
'{inqs}\n'
'You are given a few inquiries. '
'Please find matched item for each inquiry and list all answers in the given format.'
)

one_to_one_prompt_template = (
'Items\' Description:\n'
'{descs}\n'
'Inquiry:\n'
'{inqs}\n'
'You are given an inquiry. '
'Please find the best matched item and output the answer in the given format.'
)

description_template = (
'Item ID: {item_id}:\n'
'- Description: {desc}\n'
'- Position: {position}\n'
#'- Bounding Box: {bbox}\n'
)
inquiry_template = 'Inquiry ID: {inquiry_id}, Inquiry Content: {content}.\n'

def openai_matching(messages, model):
#    print(json.dumps(messages, indent=2))
#    input()
    response = openai_api.call_json(
        messages,
#        response_format=AllMatches,
        response_format=OneMatch,
        model=model, # 'gpt-4o-mini' / 'gpt-4o'
        max_tokens=1000,
    )
    response_json = response.model_dump(mode='json')
    #print(json.dumps(response_json, indent=2))
    #input()
    # fp.write(json.dumps(response_json)+'\n')
    # fp.close()
#    return response_json['matches']
    return response_json

def anthropic_matching(messages, model):
    tools = [
        {
            "name": "record_summary",
            "description": "Record summary of an image using well-structured JSON.",
#            "input_schema": AllMatches.model_json_schema(mode='validation'),
            "input_schema": OneMatch.model_json_schema(mode='validation'),
        },
    ]
    response_json = anthropic_api.call(
        messages,
        tools,
        model=model,
        max_tokens=1000,
    )
#    return response['matches']
    return response

def vllm_matching(messages, match_model):
#    schema = AllMatches.model_json_schema(mode='validation')
    schema = OneMatch.model_json_schema(mode='validation')
    response_json = vllm_api.call(
        messages,
        schema,
        model=match_model,
        max_new_tokens=1000,
        mode='llm',
    )
    return response_json

def expression_one_match(desc_list, inquiry, model='gpt-4o'):
    descriptions_text = []
    for i, pred in enumerate(desc_list):
        bbox = pred['bbox']
        desc_text = description_template.format(
            item_id = i, #int(pred['category_id']),
            desc = pred['object_info'], #json.dumps(descriptions[did-1]),
            position = [bbox[0], bbox[1]], # left-top corner or middle point +bbox[2]//2 +bbox[3]//2
            #bbox = [int(i) for i in pred['bbox'].tolist()],
        )
        descriptions_text.append(desc_text)
    descriptions_text = ''.join(descriptions_text)
#    print(descriptions_text)

    prompt = one_to_one_prompt_template.format(descs=descriptions_text, inqs=inquiry)

    messages = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': system_prompt,
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt,
                },
            ],
        },
    ]

#    print(model)
    if 'gpt-4' == model[:5]:
        matches = openai_matching(messages, model)
    elif 'claude' == model[:6]:
        matches = anthropic_matching(messages, model)
    elif 'llama' == model[:5] or 'qwen' == model[:4]:
        matches = vllm_matching(messages, model)

    if isinstance(matches, str):
        return {}

    return matches

def expression_match(desc_list, inquiries, model='gpt-4o'):
   #fp = open('matches.jsonl', 'a+')
    descriptions_text = []
    for i, pred in enumerate(desc_list):
        bbox = pred['bbox']
        desc_text = description_template.format(
            item_id = i, #int(pred['category_id']),
            desc = pred['object_info'], #json.dumps(descriptions[did-1]),
            position = [bbox[0], bbox[1]], # left-top corner or middle point +bbox[2]//2 +bbox[3]//2
            #bbox = [int(i) for i in pred['bbox'].tolist()],
        )
        descriptions_text.append(desc_text)
    descriptions_text = ''.join(descriptions_text)
#    print(descriptions_text)
    
    inquiries_text = []
    for iid, inquiry in enumerate(inquiries):
        inq_text = inquiry_template.format(
            inquiry_id=iid,
            content=inquiry,
        )
        inquiries_text.append(inq_text)
    inquiries_text = ''.join(inquiries_text)
#    print(inquiries_text)
    
    prompt = group_prompt_template.format(descs=descriptions_text, inqs=inquiries_text)
#    print(prompt)
#    input()
    messages = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': system_prompt,
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt,
                },
            ],
        },
    ]

    print(model)
    if 'gpt-4' == model[:5]:
        matches = openai_matching(messages, model)
    elif 'claude' == model[:6]:
        matches = anthropic_matching(messages, model)
    elif 'llama' == model[:5] or 'qwen' == model[:4]:
        matches = vllm_matching(messages, model)

    if isinstance(matches, str):
        return {}

    return matches


if __name__ == '__main__':
    desc_list = [
        [5, None,  (438, 346, 219, 119)],
        [6, None, (327, 193, 184, 203)],
        [7, None, (650, 316, 630, 102)],
    ]
    
    inquiries = [
        'the orange bottle',
        'the middle one',
        'bottle with a black cap',
    ]

    expression_match(desc_list, inquiries)

