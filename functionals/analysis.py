import os
import json
import numpy as np
from scipy import stats
from functionals.agent import MbtiChats
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file, debug


def get_start():
    _, _, file_name = os.walk(
        '/opt/project/KG_Assist_LLM/logs/debug').__next__()
    file_name = [name.split('_') for name in file_name]
    file_name.sort(key=lambda x: x[1])
    file_name = file_name[-1]
    start = int(file_name[1])
    file_name = "_".join(file_name)
    with open(f"/opt/project/KG_Assist_LLM/logs/debug/{file_name}", 'r') as f:
        data = json.load(f)
    for num in data:
        start = max(start, (num[0]))
    start = start + 1
    return start


@log_to_file
def mbti_analysis(start, end, dataset='kaggle'):
    result = []
    config = ModelConfig()
    data = config.mbti_data(dataset)
    mbti = MbtiChats()
    try:
        start = max(get_start(), start)
        print(f"start: {start}")
        for i, (mbti_real, task) in enumerate(data.values[start:end]):
            print("origin_task")
            print(task)
            mbti.run(task)
            result.append(
                [i+start, mbti_real, mbti.chat_result['final_mbti'], mbti.chat_result])
            debug(result, f"{dataset}_{start}_{end}")
        count = np.zeros(4)
        for _, mbti_real, mbti_predict, _ in result:
            for i in range(4):
                if mbti_real[i] == mbti_predict[i]:
                    count[i] += 1
        count /= len(result)
        return [f"{x*100:.4f}%" for x in count], stats.hmean(count), len(result), result
    except Exception as e:
        print(e)
        start += 1
        mbti_analysis(start, end, dataset)


if __name__ == "__main__":
    print(get_start())
