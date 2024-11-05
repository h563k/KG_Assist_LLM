from functional.llm_api import llm_local
from functional.setting import ModelConfig
from functional.llm_promot import semantic_analysis

if __name__ == '__main__':
    config = ModelConfig()
    model_name = config.ollama['model_name']
    for mbti_type, txt in config.mbti_data.values:
        result = semantic_analysis(txt, model_name)
        print(result)
        break

    # system_prompt = '请用中文回答问题'
    # prompt = '你知道 mbti 吗'
    # model_name = 'llama3.1:70b'
    # stream = False
    # print(llm_local(system_prompt, prompt, model_name, stream))
