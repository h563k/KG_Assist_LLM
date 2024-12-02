from openai import OpenAI
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file

config = ModelConfig()


def openai_response(system_prompt, prompt, model_types, client: OpenAI, stream=False) -> str:
    response = client.chat.completions.create(
        model=model_types,
        seed=config.llm['seed'],
        temperature=config.llm['temperature'],
        presence_penalty=config.llm['presence_penalty'],
        frequency_penalty=config.llm['frequency_penalty'],
        max_tokens=config.llm['max_tokens'],
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


@log_to_file
def llm_free(system_prompt, prompt, model_types, stream=False) -> str:
    port, token = config.llm[model_types]
    header = {
        "Content-types": "application/json",
        "Authorization": f"Bearer {token}"
    }
    client = OpenAI(base_url=f"http://192.168.28.5:{port}/v1/",
                    api_key="not used actually",
                    default_headers=header
                    )
    return openai_response(system_prompt, prompt, model_types, client, stream)


@log_to_file
def llm_local(system_prompt, prompt, model_name, stream=False) -> str:
    client = OpenAI(base_url="http://192.168.28.5:11434/v1",
                    api_key="ollama")
    return openai_response(system_prompt, prompt, model_name, client, stream)
