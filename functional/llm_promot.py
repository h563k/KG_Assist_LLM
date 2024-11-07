import re
from functional.llm_api import llm_local
from functional.setting import ModelConfig
from functional.standard_log import log_to_file

config = ModelConfig()


def data_process(txt: str):
    temp = []
    txt = txt.split('|||')
    for message in txt:
        website = re.findall('(https://\S+|http://\S+)', message)
        if not website:
            continue
        for web in website:
            message = message.replace(web, '')
        temp.append(message)
    txt = ";".join(temp)
    return txt


def promot_analysis(promot, model_name, stream=False):
    result = llm_local(system_prompt="",
                       prompt=promot,
                       model_name=model_name,
                       stream=stream)
    return result

@log_to_file
def mbti_analysis(model_name=config.ollama['model_name'], stream=False, lens=-1) -> str:
    temp = []
    for i, (mbti_type, txt) in enumerate(config.mbti_data.values):
        txt = data_process(txt)
        promot_semantic = f"""**Please provide a concise analysis of the semantic content of the following text:**

        ### Text for Analysis
        
        {txt}"""

        promot_sentiment = f"""**Please provide a concise analysis of the sentiment content of the following text:**
        
        ### Text for Analysis
        
        {txt}"""
        promot_inguistic = f"""**Please provide a concise analysis of the linguistic content of the following text:**

        ### Text for Analysis
        {txt}"""

        semantic = promot_analysis(promot_semantic, model_name, stream)
        sentiment = promot_analysis(promot_sentiment, model_name, stream)
        inguistic = promot_analysis(promot_inguistic, model_name, stream)
        promot_mbti = f"""**Predict the MBTI personality type(s) of the individual(s) based on the text. Provide only the answer without the analysis:**
        {semantic}
        {sentiment}
        {inguistic}"""
        mbti_type_predict = promot_analysis(promot_mbti, model_name, stream)
        temp.append([i, mbti_type, mbti_type_predict])
        if i == lens:
            break
    return temp
