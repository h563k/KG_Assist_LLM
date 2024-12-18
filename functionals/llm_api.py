import openai
import backoff
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file


config = ModelConfig()
mbti = config.mbti


@backoff.on_exception(backoff.expo, openai.RateLimitError)
@log_to_file
def openai_response(system_prompt, prompt, openai_type='openai_origin', stream=False) -> str:
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
