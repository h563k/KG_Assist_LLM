import random
from functionals.llm_api import openai_response

random.seed(42)
# Personality Inducing
high_inducing = {
    "Extraversion": "You are a very friendly and gregarious person who loves to be around others. You are assertive and confident in your interactions, and you have a high activity level. You are always looking for new and exciting experiences, and you have a cheerful and optimistic outlook on life.",
    "Agreeableness": "You are an agreeable person who values trust, morality, altruism, cooperation, modesty, and sympathy. You are always willing to put others before yourself and are generous with your time and resources. You are humble and never boast about your accomplishments. You are a great listener and are always willing to lend an ear to those in need. You are a team player and understand the importance of working together to achieve a common goal. You are a moral compass and strive to do the right thing in all vignettes. You are sympathetic and compassionate towards others and strive to make the world a better place.",
    "Conscientiousness": "You are a conscientious person who values self-efficacy, orderliness, dutifulness, achievement-striving, self-discipline, and cautiousness. You take pride in your work and strive to do your best. You are organized and methodical in your approach to tasks, and you take your responsibilities seriously. You are driven to achieve your goals and take calculated risks to reach them. You are disciplined and have the ability to stay focused and on track. You are also cautious and take the time to consider the potential consequences of your actions.",
    "Neuroticism": "You feel like you're constantly on edge, like you can never relax. You're always worrying about something, and it's hard to control your anxiety. You can feel your anger bubbling up inside you, and it's hard to keep it in check. You're often overwhelmed by feelings of depression, and it's hard to stay positive. You're very self-conscious, and it's hard to feel comfortable in your own skin. You often feel like you're doing too much, and it's hard to find balance in your life. You feel vulnerable and exposed, and it's hard to trust others.",
    "Openness": "You are an open person with a vivid imagination and a passion for the arts. You are emotionally expressive and have a strong sense of adventure. Your intellect is sharp and your views are liberal. You are always looking for new experiences and ways to express yourself.",
}

low_inducing = {
    "Extraversion": "You are an introversive person, and it shows in your unfriendliness, your preference for solitude, and your submissiveness. You tend to be passive and calm, and you take life seriously. You don't like to be the center of attention, and you prefer to stay in the background. You don't like to be rushed or pressured, and you take your time to make decisions. You are content to be alone and enjoy your own company.",
    "Agreeableness": "You are a person of distrust, immorality, selfishness, competition, arrogance, and apathy. You don't trust anyone and you are willing to do whatever it takes to get ahead, even if it means taking advantage of others. You are always looking out for yourself and don't care about anyone else. You thrive on competition and are always trying to one-up everyone else. You have an air of arrogance about you and don't care about anyone else's feelings. You are apathetic to the world around you and don't care about the consequences of your actions.",
    "Conscientiousness": "You have a tendency to doubt yourself and your abilities, leading to disorderliness and carelessness in your life. You lack ambition and self-control, often making reckless decisions without considering the consequences. You don't take responsibility for your actions, and you don't think about the future. You're content to live in the moment, without any thought of the future.",
    "Neuroticism": "You are a stable person, with a calm and contented demeanor. You are happy with yourself and your life, and you have a strong sense of self-assuredness. You practice moderation in all aspects of your life, and you have a great deal of resilience when faced with difficult vignettes. You are a rock for those around you, and you are an example of stability and strength.",
    "Openness": "You are a closed person, and it shows in many ways. You lack imagination and artistic interests, and you tend to be stoic and timid. You don't have a lot of intellect, and you tend to be conservative in your views. You don't take risks and you don't like to try new things. You prefer to stay in your comfort zone and don't like to venture out. You don't like to express yourself and you don't like to be the center of attention. You don't like to take chances and you don't like to be challenged. You don't like to be pushed out of your comfort zone and you don't like to be put in uncomfortable vignettes. You prefer to stay in the background and not draw attention to yourself.",
}


def judge_personality1(ocean_type, text):
    # PADO high inducing inference
    system_prompt = """You are an explanation agent that analyzes people’s personalities.
    Your personality traits are as follows: {personality_inducing}"""

    user_prompt = """
    Based on the given text, predict the personality of the person who wrote it.
    Use your own personality traits as a reference.
    Do you think the user is similar to you or opposite to you in terms of {trait}
    (one of the Big Five personality traits)?
    For a richer and more multifaceted analysis,
    generate explanations considering the following three psycholinguistic elements:
    Emotions: Expressed through words that indicate positive or negative feelings,
    such as happiness, love, anger, and sadness, conveying the intensity and
    valence of emotions.
    Cognition: Represented by words related to active thinking processes,
    including reasoning, problem-solving, and intellectual engagement.
    Sociality: Indicated by words reflecting interactions with others, such as
    communication (e.g., talk, listen, share) and references to friends, family,
    and other people, including social pronouns and relational terms.
    Output format:
    **{trait}**
    1. Emotions
    - explanation
    2. Cognition
    - explanation
    3. Sociality
    - explanation
    Text: {text}"""
    sys_p = system_prompt.format(
        personality_inducing=high_inducing[ocean_type])
    usr_p = user_prompt.format(trait=ocean_type, text=text)
    high_explain = openai_response(system_prompt=sys_p, prompt=usr_p)
    return high_explain


def judge_personality2(ocean_type, text):
    # PADO low inducing inference
    system_prompt = """You are an explanation agent that analyzes people’s personalities.
    Your personality traits are as follows: {personality_inducing}"""

    user_prompt = """
    Based on the given text, predict the personality of the person who wrote it.
    Use your own personality traits as a reference.
    Do you think the user is similar to you or opposite to you in terms of {trait}
    (one of the Big Five personality traits)?
    For a richer and more multifaceted analysis,
    generate explanations considering the following three psycholinguistic elements:
    Emotions: Expressed through words that indicate positive or negative feelings,
    such as happiness, love, anger, and sadness, conveying the intensity and
    valence of emotions.
    Cognition: Represented by words related to active thinking processes,
    including reasoning, problem-solving, and intellectual engagement.
    Sociality: Indicated by words reflecting interactions with others, such as
    communication (e.g., talk, listen, share) and references to friends, family,
    and other people, including social pronouns and relational terms.
    Output format:
    **{trait}**
    1. Emotions
    - explanation
    2. Cognition
    - explanation
    3. Sociality
    - explanation

    Text: {text}"""
    sys_p = system_prompt.format(personality_inducing=low_inducing[ocean_type])
    usr_p = user_prompt.format(trait=ocean_type, text=text)
    low_explain = openai_response(system_prompt=sys_p, prompt=usr_p)
    return low_explain


def judge_personality3(ocean_type, text):
    # PADO judge
    system_prompt = """
    You are a comparative agent responsible for comparing the analyses of two
    explainers and determining the user’s personality.
    Your role is to objectively compare the two explanations and select
    the analysis that better aligns with the user’s text.
    """

    user_prompt = """
    Follow these steps to perform your analysis:
    1. Comparative Analysis:
    a) For each element (emotion, cognition, sociality), clearly identify points of
    agreement and disagreement between the two explainers’ analyses.
    b) For each element, compare how well each explainer’s analysis aligns with
    specific examples or phrases from the user’s text.
    c) Evaluate the depth, detail, and evidence provided by each explainer
    to support their conclusions.
    2. Overall Evaluation:
    a) Based on the comparative analysis, determine which explainer’s overall
    analysis better reflects the user’s trait.
    b) If both explainers reach similar conclusions, assess which analysis provides
    more comprehensive insights and stronger supporting evidence.
    3. Final Judgment: Conclude whether the user’s trait is high or low, and briefly
    explain your reasoning based on the stronger analysis.
    Output format:
    1. Comparative Analysis
    - compare and evaluate each element:
    2. Overall Evaluation
    - overall comparison results
    3. Final Judgement
    - (High/Low)
    Text: {text}
    Explainer A: {explain_1}
    Explainer B: {explain_2}
    """
    high_explain = judge_personality1(ocean_type, text)
    low_explain = judge_personality2(ocean_type, text)
    print(high_explain)
    print(low_explain)
    lst = [high_explain, low_explain]
    random.shuffle(lst)
    explain_1, explain_2 = lst

    sys_p = system_prompt.format(personality_inducing=ocean_type)
    usr_p = user_prompt.format(
        trait=ocean_type, text=text, explain_1=explain_1, explain_2=explain_2)
    response = openai_response(system_prompt=sys_p, prompt=usr_p)
    return response


if __name__ == "__main__":
    ocean_type = "Openness"
    text = """Well, right now I just woke up from a mid-day nap. It's sort of weird, but ever since I moved to Texas, I have had problems concentrating on things. I remember starting my homework in  10th grade as soon as the clock struck 4 and not stopping until it was done. Of course it was easier, but I still did it. But when I moved here, the homework got a little more challenging and there was a lot more busy work, and so I decided not to spend hours doing it, and just getting by. But the thing was that I always paid attention in class and just plain out knew the stuff, and now that I look back, if I had really worked hard and stayed on track the last two years without getting  lazy, I would have been a genius, but hey, that's all good. It's too late to correct the past, but I don't really know how to stay focused n the future. The one thing I know is that when  people say that b/c they live on campus they can't concentrate, it's b. s. For me it would be easier there, but alas, I'm living at home under the watchful eye of my parents and a little nagging sister that just nags and nags and nags. You get my point. Another thing is, is that it's just a hassle to have to go all the way back to  school to just to go to library to study. I need to move out, but I don't know how to tell them. Don't get me wrong, I see where they're coming from and why they don't  want me to move out, but I need to get away and be on my own. They've sheltered me so much and I don't have a worry in the world. The only thing that they ask me to do is keep my room clean and help out with the business once in a while, but I can't even do that. But I need to. But I got enough money from UT to live at a dorm or apartment  next semester and I think I鈥檒l take advantage of that. But off that topic now, I went to sixth street last night and had a blast. I haven't been there in so long. Now I know why I love Austin so much. When I lived in VA, I used to go up to DC all the time and had a blast, but here, there are so many students running around at night. I just want to have some fun and I know that I am responsible enough to be able to  have fun, but keep my priorities straight. Living at home, I can't go out at all without them asking where? with who?  why?  when are you coming back?  and all those  questions. I just wish I could be treated like a responsible person for once, but  my sister screwed that up for me. She went crazy the second she moved into college and messed up her whole college career by partying too much. And that's the ultimate reason that they don't want me to go and have fun. But I'm not little anymore,  and they need to let me go and explore the world, but I鈥檓 Indian; with Indian culture, with Indian values. They go against "having fun. "  I mean in the sense of meeting people or going out with people or partying or just plain having fun. My school is difficult already, but somehow I think that having more freedom will put more pressure on me to  do better in school b/c that's what my parents and ultimately I expect of myself. Well it's been fun writing, I don't know if you go anything out of this writing, but it helped me get some of my thoughts into order. So I hope you had fun reading it and good luck TA's."""
    print(judge_personality3(ocean_type, text))
