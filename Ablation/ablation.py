import re
from sklearn.metrics import f1_score


def get_mbti_predict(circle_chats: str):
    starts = ['1.', '2.', '3.', '4.']
    circle_chats = circle_chats.split("\n\n")
    temp = ""
    for circle_chat in circle_chats:
        circle_chat = circle_chat.strip()
        come_on = False
        for start in starts:
            if start in circle_chat:
                come_on = True
        if not come_on:
            continue
        circle_chat = circle_chat.split("\n")[0]
        mbti_predict = re.findall('E|I|S|N|T|F|J|P', circle_chat)
        if mbti_predict:
            temp += mbti_predict[-1]
    if len(temp) != 4:
        print(circle_chats)
        return
    return temp


def get_first_chat(datas: list):
    full_1 = [[], [], [], []]
    for data in datas:
        real = data[1]
        data = data[-1]
        first_chats = data['first_chats']
        for i in range(3):
            llm_response = first_chats[i]['content']
            llm_response = get_mbti_predict(llm_response)
            full_1[i].append(list(llm_response))
        full_1[3].append(list(real))
    print(full_1)

"""
mbti_real
[['I', 'N', 'T', 'J'], ['I', 'S', 'T', 'P'], ['I', 'N', 'T', 'P']]
mbti_predict
[['I', 'N', 'T', 'J'], ['I', 'N', 'T', 'J'], ['I', 'S', 'F', 'P']]
"""
def avgmaf1(mbti_real, mbti_predict):
    lens = len(mbti_real)
    res = [[0 for _ in range(lens)] for _ in range(8)]
    for j in range(lens):
        for k in range(4):
            res[k][j] = mbti_real[j][k]
            res[k+4][j] = mbti_predict[j][k]
    temp = []
    show = []
    score = 0
    for i in range(4):
        show.append(res[i][-10:])
        show.append(res[i+4][-10:])
        macro_f1 = f1_score(res[i], res[i+4], average='macro')
        score += macro_f1
        temp.append(macro_f1)
    print(temp, score/4)
    return score
