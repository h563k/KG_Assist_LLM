from functionals.system_config import ModelConfig
from functionals.agent import MbtiChats

if __name__ == '__main__':
    config = ModelConfig()
    data = config.mbti_data
    mbti = MbtiChats(max_round=3, nums="two", openai_type="openai_hk")
    task = data.iloc[0, 1]
    print(task)
    mbti.run(task)
