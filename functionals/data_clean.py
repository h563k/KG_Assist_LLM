import os
import json
import re
import time
from functionals.llm_api import openai_response
from functionals.system_config import ModelConfig

config = ModelConfig()
mbti = config.mbti

def data_process(task: str, cutoff=mbti['cutoff']):
    temp = []
    task = task.split('|||')
    for message in task:
        website = re.findall('(https://\S+|http://\S+)', message)
        if not website:
            temp.append(message)
            continue
        for web in website:
            message = message.replace(web, '')
        temp.append(message) if message else None
    txt = "\n".join(temp)
    deepclean_cutoff = cutoff * 10
    # TODO目前单纯通过文本去判断属于一个简易的判断, 这个不能作为一个非常合理的解释, 后期看看能不能给出更科学的判断依据. 比如存在大量无效文本的时候开启过滤
    deepclean = mbti['deepclean']
    # bert筛选，先进行句子拆分
    sentences = []
    for contents in txt.split('\n'):
        bodys = contents.split('. ')
        for body in bodys:
            test = [x for x in body.split(' ') if x]
            if len(test) <= 5:
                continue
            body = ' '.join(test)
            sentences.append(body)
    count = 0
    process = []
    # 不进行bert筛选，直接长度截断
    if not deepclean:
        for message in sentences:
            process.append(message)
            count += len(message)
            if count > cutoff:
                break
        return "\n".join(process)
    # bert筛选
    deep_clean_dict_path = f'{config.file_path}/data/samples/deep_clean.json'
    if os.path.exists(deep_clean_dict_path):
        with open(deep_clean_dict_path, 'r') as f:
            deep_clean_dict = json.load(f)
    else:
        deep_clean_dict = {}
    input_count = 0
    for message in sentences:
        input_count += len(message)
        system_prompt = "Please read the following content and determine if it involves any MBTI personality traits and any character characteristics,just respond with a simple 'Yes' or 'No'"
        if deep_clean_dict.get(message):
            response = deep_clean_dict[message]
            if response:
                print('deep clean dict')
        else:
            response = openai_response(system_prompt, message)
        if mbti['openai_type'] == 'ollama':
            time.sleep(0.1)
        response = response.split('None')[0]
        print({"message": message, "response": response})
        if "YES" in response.upper():
            is_mbti = 1
            deep_clean_dict[message] = response
        elif "NO" in response.upper():
            is_mbti = 0
            deep_clean_dict[message] = response
        else:
            is_mbti = 0
        if is_mbti:
            process.append(message)
            count += len(message)
        if count > cutoff or input_count > deepclean_cutoff:
            break
    print(f'deep clean count: {input_count}')
    # 一次性将更新后的字典写入文件
    with open(deep_clean_dict_path, 'w') as f:
        json.dump(deep_clean_dict, f, indent=4)

    txt = "\n".join(process)
    return txt
