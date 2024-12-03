from functionals.agent import MbtiChats
from functionals.system_config import ModelConfig



config = ModelConfig()
data = config.mbti_data
mbti = MbtiChats(max_round=2,nums="three",openai_type="openai_hk")