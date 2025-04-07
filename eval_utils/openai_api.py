import os

import tiktoken
import time
from openai import OpenAI

if os.path.exists('eval_utils/keys/openai_key'):
    key_path = 'eval_utils/keys/openai_key'
elif os.path.exists('keys/openai_key'):
    key_path = 'keys/openai_key'
with open(key_path, 'r') as f:
    api_key = f.readline().strip()

if os.path.exists('eval_utils/keys/openai_org_id'):
    org_path = 'eval_utils/keys/openai_org_id'
elif os.path.exists('keys/openai_org_id'):
    org_path = 'keys/openai_org_id'
with open(org_path, 'r') as f:
    organization = f.readline().strip()

client = OpenAI(api_key=api_key, organization=organization)

def call_json(messages, response_format=None, model='gpt-3.5-turbo-0125', max_tokens=300):
#    tokenizer = tiktoken.encoding_for_model(model)
#    print(json.dumps(messages))

    while True:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                response_format=response_format
            )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))
#            import json
#            print(json.dumps(messages, indent=2))
            input()

    message = response.choices[0].message
    if message.parsed:
        prediction = message.parsed
    else:
        prediction = message.refusal

    return prediction

