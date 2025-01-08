import re
from functionals.llm_api import openai_response


def data_process(task: str, deepclean=True, cutoff=3500):
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
    for message in sentences:
        system_prompt = "Please read the following content and determine if it involves any MBTI personality traits and any character characteristics,just respond with a simple 'Yes' or 'No'"
        response = openai_response(system_prompt, message)
        response = response.split('None')[0]
        print({"message": message, "response": response})
        if "YES" in response.upper():
            is_mbti = 1
        elif "NO" in response.upper():
            is_mbti = 0
        else:
            is_mbti = 0
        if is_mbti:
            process.append(message)
            count += len(message)
        if count > cutoff:
            break
    txt = "\n".join(process)
    return txt
