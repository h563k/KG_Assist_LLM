import re
from functional.llm_api import llm_local
from functional.setting import ModelConfig


config = ModelConfig()



def data_process(txt: str):
    temp = []
    txt = txt.split('|||')
    for message in txt:
        website = re.findall('(https://\S+|http://\S+)', message)
        if website:
            for web in website:
                message = message.replace(web, '')
        temp.append(message)
    txt = ";".join(temp)
    return txt


def semantic_analysis(txt, model_name, stream=False):
    txt = data_process(txt)
    promot = f"""Please analyze the following text for its semantic content, focusing on the following aspects:

1. Emotional Tone: Identify the overall emotional tone of the text. Is it positive, negative, neutral, or mixed? Provide examples from the text to support your analysis.
2. Themes and Topics: List the main themes and topics discussed in the text. How do these topics relate to each other?
3. Personality Traits: Based on the content, what personality traits can be inferred about the author? Consider using established personality models (e.g., Big Five, Myers-Briggs) as a framework.
4. Language Use: Analyze the language used, including any idioms, slang, or technical terms. What does this suggest about the author's background or intended audience?
5. Cultural References: Identify any cultural references (e.g., movies, books, historical events) and explain their significance in the context of the text.
6. Logical Structure: Describe the logical structure of the text. Is it organized coherently? Are there any contradictions or inconsistencies?
7. Author’s Intent: What seems to be the author’s intent in writing this text? Is there a specific message or goal they are trying to convey?
Text to Analyze:
{txt}
    """
    result = llm_local(system_prompt="",
                       prompt=promot,
                       model_name=model_name,
                       stream=stream)
    return result

# for mbti_type, txt in config.mbti_data.values:
#     result = semantic_analysis(txt)
#     print(result)
#     break