import openai
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

config = ModelConfig()
mbti = config.mbti


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
@log_to_file
def openai_response(system_prompt, prompt, openai_type='openai_origin', stream=True) -> str:
    openai.api_key = config.OpenAI[openai_type]['api_key']
    openai.base_url = config.OpenAI[openai_type]['base_url']
    response = openai.chat.completions.create(
        model=mbti['model'],
        seed=config.OpenAI['seed'],
        temperature=config.OpenAI['temperature'],
        presence_penalty=config.OpenAI['presence_penalty'],
        frequency_penalty=config.OpenAI['frequency_penalty'],
        max_tokens=config.OpenAI['max_tokens'],
        stream=stream,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    if stream:
        full_response = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            full_response += str(delta)
        return full_response
    else:
        return response.choices[0].message.content

# TODO 目前来看7b模型训练效果不太理想，待进一步研究，目前先不考虑


def llama_factory_api(prompt):
    # change to your custom port
    system_prompt = "Please read the following content and determine if it involves any MBTI personality traits and any character characteristics,just respond with a simple 'Yes' or 'No'"
    client = openai.OpenAI(
        api_key="0",
        base_url="http://127.0.0.1:8000/v1",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    result = client.chat.completions.create(messages=messages, model="test")
    return result.choices[0].message.content


def bert_api(input_text: str):
    # 加载模型和分词器
    model_path = "/opt/project/KG_Assist_LLM/data/bert_train/model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 对输入文本进行分词
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs.to(device)
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # 返回预测结果
    return predictions.item()
