import os
import json
from sklearn.metrics import f1_score


def F1ScoreCount(path: str):
    _, _, files = os.walk(path).__next__()
    datas = {}
    temp = []
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            josn_data = json.load(f)
            temp.extend(josn_data)
    lens = len(temp)
    for value in temp:
        datas[value[0]] = value[1:]
    res = [[0 for _ in range(lens)] for _ in range(8)]
    for j, (_, data) in enumerate(datas.items()):
        mbti_real = data[0]
        mbti_predict = data[1]
        for i in range(4):
            res[i][j] = mbti_real[i]
            res[i+4][j] = mbti_predict[i]
    temp = []
    show = []
    score = 0
    for i in range(4):
        show.append(res[i][-10:])
        show.append(res[i+4][-10:])
        macro_f1 = f1_score(res[i], res[i+4], average='macro')
        score += macro_f1
        temp.append(macro_f1)
    return temp, score/4, lens
