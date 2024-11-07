from functional.llm_promot import mbti_analysis

if __name__ == '__main__':
    # mbti_analysis(model_name='qwen2.5:72b-instruct-q4_0')
    result = mbti_analysis(lens=10)
    print(result)

