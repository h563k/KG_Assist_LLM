import re
from functionals.llm_api import bert_api


def data_process(txt: str, deepclean=True, cutoff=3500):
    temp = []
    txt = txt.split('|||')
    for message in txt:
        website = re.findall('(https://\S+|http://\S+)', message)
        if not website:
            temp.append(message)
            continue
        for web in website:
            message = message.replace(web, '')
        temp.append(message) if message else None
    txt = "\n".join(temp)
    if not deepclean:
        return txt
    # bert筛选，先进行句子拆分
    temp1 = []
    for contents in txt.split('\n'):
        bodys = contents.split('. ')
        for body in bodys:
            if len(body.split(' ')) <= 5:
                continue
            temp1.extend(bodys)
    # 按长度截断
    count = 0
    temp2 = []
    for i, message in enumerate(temp1):
        is_mbti = bert_api(message)
        if is_mbti:
            temp2.append(message)
            count += len(message)
        if count > cutoff:
            break
    txt = "\n".join(temp2)
    return txt
