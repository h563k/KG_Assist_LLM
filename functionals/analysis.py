import os
import json
import time
import numpy as np
from scipy import stats
from functionals.agent import MbtiChats
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file, debug

config = ModelConfig()


def get_start():
    _, _, file_names = os.walk(
        f'{config.file_path}/logs/debug').__next__()
    if not file_names:
        return 0
    start = 0
    for file_name in file_names:
        with open(f"{config.file_path}/logs/debug/{file_name}", 'r') as f:
            data = json.load(f)
        start = max(start, data[-1][0])
    start = start + 1
    return start


@log_to_file
def mbti_analysis(start, end, dataset='kaggle', type=0):
    result = []

    data = config.mbti_data(dataset)
    try:
        start = max(get_start(), start)
        print(f"start: {start}")
        for i, (mbti_real, task) in enumerate(data.values[start:end]):
            print("origin_task")
            mbti = MbtiChats()
            if type == 0:
                mbti.run(task)
            elif type == 4:
                mbti.run_single(task)
            elif type == 6:
                mbti.run_without_vote(task)
            final_mbti = mbti.chat_result['final_mbti']
            if final_mbti[0] in "EI" and final_mbti[1] in "SN" and final_mbti[2] in "TF" and final_mbti[3] in "JP":
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
        time.sleep(10)
        start += 1
        mbti_analysis(start, end, dataset)


if __name__ == "__main__":
    print(get_start())
