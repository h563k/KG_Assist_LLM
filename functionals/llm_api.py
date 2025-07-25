import openai
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


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
@log_to_file
def openai_response(system_prompt, prompt, openai_type=mbti['openai_type'], stream=True) -> str:
    openai_config = config.OpenAI
    api_key = openai_config[openai_type]['api_key']
    base_url = openai_config[openai_type]['base_url']
    if base_url.endswith('/v1') and openai_type == 'deepseek':
        base_url = base_url.strip('/v1')
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=openai_config[openai_type]['model'],
        seed=openai_config['seed'],
        temperature=openai_config['temperature'],
        presence_penalty=openai_config['presence_penalty'],
        frequency_penalty=openai_config['frequency_penalty'],
        max_tokens=openai_config['max_tokens'],
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


