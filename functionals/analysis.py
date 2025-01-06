import numpy as np
from scipy import stats
from functionals.agent import MbtiChats
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file


@log_to_file
def mbti_analysis(start, end, dataset='kaggle'):
    try:
        config = ModelConfig()
        data = config.mbti_data[dataset]
        mbti = MbtiChats()
        result = [] 
        count = 0
        for i, (mbti_real, task) in enumerate(data.values[start:end]):
            count += 1
            mbti.run(task)
            result.append([i, mbti_real, mbti.chat_result['final_mbti']])
        count = np.zeros(4)
        for _, mbti_real, mbti_predict in result:
            for i in range(4):
                if mbti_real[i] == mbti_predict[i]:
                    count[i] += 1
        count /= len(result)
        return [f"{x*100:.4f}%" for x in count], stats.hmean(count), len(result)
    except Exception as e:
        return result, count, e
