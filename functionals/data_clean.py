import re
from functionals.llm_api import bert_api


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
    # 不进行bert筛选，直接长度截断
    if not deepclean:
        count = 0
        process = []
        for message in sentences:
            process.append(message)
            count += len(message)
            if count > cutoff:
                break
        return "\n".join(process)
    # bert筛选
    count = 0
    bert_process = []
    for message in sentences:
        is_mbti = bert_api(message)
        if is_mbti:
            bert_process.append(message)
            count += len(message)
        if count > cutoff:
            break
    txt = "\n".join(bert_process)
    return txt
