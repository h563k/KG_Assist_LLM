from functional.llm_api import llm_local

if __name__ == '__main__':
    system_prompt = 'You are a helpful assistant.'
    prompt = '你知道 mbti 吗'
    model_name = 'qwen2.5:32b'
    stream = False
    print(llm_local(system_prompt, prompt, model_name, stream))
