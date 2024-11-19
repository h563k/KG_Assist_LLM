# from functionals.llm_promot import mbti_analysis
from functionals.llm_api import llm_free

if __name__ == '__main__':
    # mbti_analysis(model_name='qwen2.5:72b-instruct-q4_0')
    # result = mbti_analysis(lens=10)
    # print(result)
    res = llm_free(system_prompt="", prompt="你好", model_types='qwen')
    print(res)

