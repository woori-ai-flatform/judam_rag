import os, json, random, openai, re
from jinja2 import Template

api_key = 'apikey'
client = openai.OpenAI(api_key=api_key)

def load_from_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        template_content = file.read()

    return Template(template_content)

def img_prompt_json(id, inst, prompt, url, model_id='gpt-4o-mini', max_tokens=1000):
    res_json =  {'custom_id': id,
                 'method': 'POST',
                 'url': '/v1/chat/completions',
                 'body': {'model': model_id,
                         'messages': [{'role': 'system',
                                       'content': inst},
                                       {'role': 'user',
                                       'content': [{'type': 'text',
                                                    'text': prompt},
                                                   {'type': 'image_url',
                                                    'image_url': {'url': url}}]}],
                         'max_tokens': max_tokens}}
    return res_json

def prompt_json(id, inst, prompt, model_id='gpt-4o-mini', max_tokens=1000):
    res_json =  {'custom_id': id,
                 'method': 'POST',
                 'url': '/v1/chat/completions',
                 'body': {'model': model_id,
                         'messages': [{'role': 'system',
                                       'content': inst},
                                       {'role': 'user',
                                       'content': prompt}],
                         'max_tokens': max_tokens}}
    return res_json

def step_01_prompt_json(img_url_lst, prompt_path, role, task_type, level_lst, data_count, few_shot_k=2, model_id='gpt-4o', max_tokens=3000):
    sys_prompt_template = load_template(prompt_path['sys_prompt_path'])
    usr_prompt_template = load_template(prompt_path['usr_prompt_path'])
    few_shot_prompt_template = load_template(prompt_path['few_shot_prompt_path'])
    few_shot_lst = load_from_jsonl(prompt_path['few_shot_lst_path'])

    prompt_data = []
    for count in range(data_count):
        for page_num, url in img_url_lst:
            for lv, level in level_lst:
                few_shot = '\n\n'.join(random.sample(few_shot_lst, k=few_shot_k))
                few_shot_prompt = few_shot_prompt_template.render(few_shot_sample=few_shot)
                sys_prompt = sys_prompt_template.render(few_shot=few_shot_prompt, role=role, task_type=task_type, level=level)
                usr_prompt = usr_prompt_template.render(n_datasets=5)
                data_dict = {
                    'custom_id': '{}_{}_{}'.format(page_num, lv, count),
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {'model': model_id,
                            'messages': [{'role': 'system',
                                          'content': sys_prompt},
                                          {'role': 'user',
                                          'content': [{'type': 'text',
                                                      'text': usr_prompt},
                                                      {'type': 'image_url',
                                                      'image_url': {'url': url}}]}],
                            'max_tokens': max_tokens}
                }
                prompt_data.append(data_dict)
    return prompt_data

def step_02_prompt_json(qa_data, prompt_path, model_id='gpt-4o', max_tokens=3000):
    sys_prompt_template = load_template(prompt_path['sys_prompt_path'])
    usr_prompt_template = load_template(prompt_path['usr_prompt_path'])

    prompt_data = []
    for data in qa_data:
        context = data['context']
        question = data['question']
        sys_prompt = sys_prompt_template.render()
        usr_prompt = usr_prompt_template.render(context=context, question=question)
        data_dict = {
            'custom_id': data['id'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {'model': model_id,
                    'messages': [{'role': 'system',
                                  'content': sys_prompt},
                                  {'role': 'user',
                                  'content': usr_prompt}],
                    'max_tokens': max_tokens}
        }
        prompt_data.append(data_dict)
    return prompt_data

def doc_step_01_prompt_json(doc_data, prompt_path, role, task_type, level_lst, few_shot_k=2, model_id='gpt-4o', max_tokens=3000):
    sys_prompt_template = load_template(prompt_path['sys_prompt_path'])
    usr_prompt_template = load_template(prompt_path['usr_prompt_path'])
    few_shot_prompt_template = load_template(prompt_path['few_shot_prompt_path'])
    few_shot_lst = load_from_jsonl(prompt_path['few_shot_lst_path'])

    prompt_data = []
    n = 0
    for context in doc_data:
        for lv, level in level_lst:
            few_shot = '\n\n'.join(random.sample(few_shot_lst, k=few_shot_k))
            few_shot_prompt = few_shot_prompt_template.render(few_shot_sample=few_shot)
            sys_prompt = sys_prompt_template.render(few_shot=few_shot_prompt, role=role, task_type=task_type, level=level)
            usr_prompt = usr_prompt_template.render(n_datasets=5, context=context)
            data_dict = {
                'custom_id': 'doc_qa_{}'.format(n),
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {'model': model_id,
                        'messages': [{'role': 'system',
                                      'content': sys_prompt},
                                      {'role': 'user',
                                      'content': usr_prompt}],
                        'max_tokens': max_tokens}
            }
            prompt_data.append(data_dict)
            n += 1
    return prompt_data



def gpt_batch_request(filename, client=client):
    batch_input_file = client.files.create(
        file=open(filename, "rb"),
        purpose="batch"
    )

    completion = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print('batch_id : {}'.format(completion.id))
    return completion.id

def gpt_batch_status(id, client=client):
    print(client.batches.retrieve(id).status)
    print(client.batches.retrieve(id).request_counts)

def gpt_result_file_save(id, filename, client=client):
    if client.batches.retrieve(id).status not in ['completed']:
        print('Not yet complete')
        return None
    output_file_id = client.batches.retrieve(id).output_file_id
    result = client.files.content(output_file_id).content
    with open(filename, 'wb') as f:
        f.write(result)
    print('complete and save')