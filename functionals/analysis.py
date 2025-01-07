import json
from datetime import datetime
import numpy as np
from scipy import stats
from functionals.agent import MbtiChats
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file


@log_to_file
def mbti_analysis(start, end, dataset='kaggle'):
    result = []
    config = ModelConfig()
    data = config.mbti_data(dataset)
    mbti = MbtiChats()
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    try:
        for i, (mbti_real, task) in enumerate(data.values[start:end]):
            mbti.run(task)
            result.append(
                [i, mbti_real, mbti.chat_result['final_mbti'], mbti.chat_result])
        count = np.zeros(4)
        for _, mbti_real, mbti_predict, _ in result:
            for i in range(4):
                if mbti_real[i] == mbti_predict[i]:
                    count[i] += 1
        count /= len(result)
        with open(f"/opt/project/KG_Assist_LLM/logs/{now}_{dataset}.josn", "w") as f:
            json.dump(result, f, indent=4)
        return [f"{x*100:.4f}%" for x in count], stats.hmean(count), len(result)
    except Exception as e:
        print(e)
        with open(f"/opt/project/KG_Assist_LLM/logs/{now}_{dataset}.josn", "w") as f:
            json.dump(result, f, indent=4)
        return result
