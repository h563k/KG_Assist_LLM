import openai
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

config = ModelConfig()
mbti = config.mbti


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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
