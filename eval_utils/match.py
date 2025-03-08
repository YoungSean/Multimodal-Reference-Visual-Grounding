import json

from eval_utils import openai_api
from pydantic import BaseModel

class Match(BaseModel):
    inquiry_id: int
    item_id: int

class AllMatches(BaseModel):
    matches: list[Match]

system_prompt = (
'You are an expert at matching information. '
'You will be given a list of items\'s descriptions and inquiries. '
'Please match items with inquiries and convert matches into the given structure.'
)

prompt_template = (
'Items\' Description:\n'
'{descs}\n'
'Inquiries:\n'
'{inqs}\n'
'You are given a few inquiries. '
'Please find matched item for each inquiry and list all answers in the given format.'
)

description_template = (
'Item ID: {item_id}:\n'
'- Description: {desc}\n'
'- Position: {position}\n'
#'- Bounding Box: {bbox}\n'
)
inquiry_template = 'Inquiry ID: {inquiry_id}, Inquiry Content: {content}.\n'

def expression_match(desc_list, inquiries):
    fp = open('matches.jsonl', 'a+')
    descriptions_text = []
    for i, pred in enumerate(desc_list):
        desc_text = description_template.format(
            item_id = i, #int(pred['category_id']),
            desc = pred['object_info'], #json.dumps(descriptions[did-1]),
            position = [int(i) for i in pred['bbox'][:2].tolist()], # the top left corner
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
    
    prompt = prompt_template.format(descs=descriptions_text, inqs=inquiries_text)
    print(prompt)
    #input()
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
    response = openai_api.call_json(
        messages,
        response_format=AllMatches,
        model='gpt-4o-mini',
        max_tokens=1000,
    )
    response_json = response.model_dump(mode='json')
    print(json.dumps(response_json, indent=2))
    input()
    fp.write(json.dumps(response_json)+'\n')
    fp.close()
    return response_json['matches']


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

