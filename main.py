from functionals.system_config import ModelConfig
from functionals.agent import MbtiChats

if __name__ == '__main__':
    config = ModelConfig()
    data = config.mbti_data
    task = data.iloc[0, 1]
    mbti = MbtiChats(task=task, nums='three', max_round=3)
