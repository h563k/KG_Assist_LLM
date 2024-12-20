import openai
import random
import time
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file


config = ModelConfig()
mbti = config.mbti
# define a retry decorator


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
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
