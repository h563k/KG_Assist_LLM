import autogen
from functionals.setting import ModelConfig
from functionals.llm_promot import data_process
from functionals.standard_log import log_to_file

"""
完整思路
### 首先这里我们需要以下几个agent:
1. planner: 确定完成任务所需的相关信息。
2. semantic专家: 分析文本的语义。
3. sentiment专家: 分析文本的情感。
4. linguistic专家: 分析文本的语法和词汇。
5. user_proxy: 分发具体的任务.
- 最后,我们记录每个专家的性格预测, 将这个预测结果作为最终的答案.
如果我们只是预测结果, 那么这里就已经可以了, 如果需要进一步的去交给小模型预测, 可以用这个列表做为输出交给小模型.


"""


config = ModelConfig()
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "base_url": "https://api.openai-hk.com/v1",
        "api_type": "openai",
        "api_key": config.openaiHk['api_key'],
        "temperature": 0.2,
    }
]
llm_config = {"config_list": config_list}

user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="Give the task and send instructions for various experts to discuss and predict the user's MBTI personality."
,
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)

planner = autogen.ConversableAgent(
    name="Planner",
    system_message="""Assign the task to experts in semantics, sentiment analysis, 
and linguistics to analyze and discuss the user information together. 
Each expert should contribute three times and provide their MBTI personality prediction for the user after the discussions have concluded.
You can conclude the conversation by saying: Thank you all for participating.
""",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
    human_input_mode="NEVER",
)


for i, (mbti_type, txt) in enumerate(config.mbti_data.values):
    txt = data_process(txt)
    task = f"""
    Here are some statements crawled from a user; please predict this user's MBTI personality based on these statements, and you can discuss among yourselves:
    {txt}
    """

    break
