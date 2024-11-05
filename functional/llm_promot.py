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


def promot_analysis(promot, model_name, stream=False):
    result = llm_local(system_prompt="",
                       prompt=promot,
                       model_name=model_name,
                       stream=stream)
    return result


def mbti_analysis(stream=False) -> str:

    model_name = config.ollama['model_name']
    for mbti_type, txt in config.mbti_data.values:
        txt = data_process(txt)
        promot_semantic = f"""**Please analyze the following text for its semantic content, focusing on the following aspects:**

        1. Emotional Tone: Identify the overall emotional tone of the text. Is it positive, negative, neutral, or mixed? Provide examples from the text to support your analysis.
        2. Themes and Topics: List the main themes and topics discussed in the text. How do these topics relate to each other?
        3. Personality Traits: Based on the content, what personality traits can be inferred about the author? Consider using established personality models (e.g., Big Five, Myers-Briggs) as a framework.
        4. Language Use: Analyze the language used, including any idioms, slang, or technical terms. What does this suggest about the author's background or intended audience?
        5. Cultural References: Identify any cultural references (e.g., movies, books, historical events) and explain their significance in the context of the text.
        6. Logical Structure: Describe the logical structure of the text. Is it organized coherently? Are there any contradictions or inconsistencies?
        7. Author’s Intent: What seems to be the author’s intent in writing this text? Is there a specific message or goal they are trying to convey?

        ### Text for Analysis
        
        -{txt}
        
        ### Output Format:
        
        - Emotional Tone: [Positive/Negative/Neutral/Mixed]
        - Themes and Topics: [List themes and topics here]
        - Personality Traits: [List personality traits here]
        - Language Use: [List language use here]
        - Cultural References: [List cultural references here]
        - Logical Structure: [Describe logical structure here]
        - Author’s Intent: [Describe author’s intent here]"""

        promot_sentiment = f"""**Please analyze the sentiment of the following text and provide a summary of the overall emotional tone. Consider the following aspects:**

        1. Positive Sentiment: Look for expressions of happiness, satisfaction, achievement, or optimism.
        2. Negative Sentiment: Identify any signs of frustration, disappointment, sadness, or pessimism.
        3. Neutral Sentiment: Note any statements that do not clearly express positive or negative emotions.
        4. Mixed Sentiment: Highlight any parts of the text that contain conflicting emotions.
        5. Additionally, comment on the subject matter and any particular themes or topics that are prevalent in the text. 
        
        ### Text for Analysis
        
        {txt}

        ### Output Format:

        - Overall Sentiment: [Positive/Negative/Neutral/Mixed]
        - Summary of Emotional Tone: [Your summary here]
        - Key Themes: [List key themes here]
        - Specific Comments: [Any additional comments or observations]
        """
        promot_inguistic = f"""**Please conduct a linguistic analysis of the following text, which is a series of posts from a user on a social media platform. The user seems to identify with the ENTP personality type and discusses various topics including personal relationships, intelligence, and online behavior.**

        ### Analysis Requirements:
        1. Language Style: Identify and describe the overall language style used by the author (e.g., formal, informal, colloquial).
        2. Vocabulary Choice: Note any specific vocabulary choices that stand out, particularly those that might reflect the author's personality or the context of the discussion.
        3. Sentiment Analysis: Analyze the emotional tone of the text (e.g., positive, negative, neutral) and provide examples of phrases or words that contribute to this tone.
        4. Rhetorical Devices: Identify any rhetorical devices used in the text (e.g., irony, hyperbole, metaphor) and explain how they function within the context.
        5. Personal References: Comment on the frequency and nature of personal references (e.g., "my girlfriend," "I over think things").
        6. Humor and Playfulness: Evaluate the use of humor and playfulness in the text, and how it contributes to the overall tone and engagement.
        7. Cultural References: Note any cultural references (e.g., Sherlock Holmes quote) and discuss their relevance to the text.
        8. Personality Indicators: Based on the linguistic features, provide insights into the author's personality traits, especially as they relate to the ENTP personality type.

        ### Text for Analysis
        {txt}
        
        ### Output Format:
        - Language Style: [Formal/Informal/Colloquial]
        - Vocabulary Choice: [List vocabulary choices here]
        - Sentiment Analysis: [Overall sentiment: Positive/Negative/Neutral]
        - Examples of Sentiment Contribution: [Phrases or words that contribute to the sentiment]
        - Rhetorical Devices: [List rhetorical devices used here]
        - Personal References: [Frequency and nature of personalreferences]
        """
        promot_mbti = f""""""
        semantic = promot_analysis(promot_semantic, model_name, stream)
        sentiment = promot_analysis(promot_sentiment, model_name, stream)
        inguistic = promot_analysis(promot_inguistic, model_name, stream)

        break
