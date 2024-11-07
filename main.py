from functional.llm_api import llm_local
from functional.setting import ModelConfig
from functional.llm_promot import mbti_analysis

if __name__ == '__main__':
    # mbti_analysis(model_name='qwen2.5:72b-instruct-q4_0')
    mbti_analysis()

    # system_prompt = '请用中文回答问题'
    # prompt = '你知道 mbti 吗'
    # model_name = 'llama3.1:70b'
    # stream = False
    # print(llm_local(system_prompt, prompt, model_name, stream))
