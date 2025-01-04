import os
import re
from autogen import initiate_chats, ConversableAgent
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file
from functionals.data_clean import data_process
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


config = ModelConfig()
mbti = config.mbti

# TODO 考虑增加对于大量文本输入先使用小模型进行过滤


class MbtiChats:
    def __init__(self, max_round=mbti['max_round'], nums=mbti['nums'], openai_type=mbti['openai_type'], model=mbti['model'], deepclean=mbti['deepclean'], cutoff=mbti['cutoff']) -> None:
        """
        :param max_round: Number of rounds for free discussion among three agents. 
        专家讨论的轮数
        :param nums: The maximum number of MBTI personality types each agent is allowed to predict, suggested using words rather than Arabic numerals d. 
        专家每轮预测的性格数量上限
        :param openai_type: The type of OpenAI API to use, hk or origin.
        代理切换
        :param model: openai model type, gpt-3.5-turbo or gpt-4
        模型选择
        """
        assert model in ['gpt-3.5-turbo',
                         'gpt-4'], "Please select the correct model type."
        self.model = model
        self.llm_config = self.env_init(openai_type)
        self.nums = nums
        self.max_round = max_round
        self.chat_result = {}
        self.agent_dict = {
            "user_proxy": self.user_proxy(),
            "Semantic": self.create_agent("Semantic"),
            "Sentiment": self.create_agent("Sentiment"),
            "Linguistic": self.create_agent("Linguistic"),
            "Commentator": self.commentator()
        }
        self.deepclean = deepclean
        self.cutoff = cutoff

    def env_init(self, openai_type) -> None:
        os.environ['http_proxy'] = config.OpenAI['proxy']
        os.environ['https_proxy'] = config.OpenAI['proxy']
        os.environ['ftp_proxy'] = config.OpenAI['proxy']
        os.environ['no_proxy'] = '127.0.0.1,localhost'
        os.environ['HTTP_PROXY'] = config.OpenAI['proxy']
        os.environ['HTTPS_PROXY'] = config.OpenAI['proxy']
        os.environ['FTP_PROXY'] = config.OpenAI['proxy']
        os.environ['NO_PROXY'] = '127.0.0.1,localhost'
        openai_config = config.OpenAI['openai_origin']
        openai_config = config.OpenAI[openai_type]
        config_list = [
            {
                "model": self.model,
                "api_type": "openai",
                "base_url": openai_config['base_url'],
                "api_key": openai_config['api_key'],
            }
        ]
        llm_config = {"config_list": config_list, }
        return llm_config
    # TODO 增加对大量文本的过滤处理。或者调用 bert 或者调用小模型，注意比较下效果

    @staticmethod
    def chat_unit(sender, recipient, message):
        return {
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "summary_method": "reflection_with_llm",
            "max_turns": 1,
            "clear_history": True
        }

    def user_proxy(self):
        agent = ConversableAgent(
            name="MessageForwarderAgent",
            system_message="You are a message forwarder, and your task is to forward the received messages unaltered to the next recipient.",
            llm_config=False,  # 不使用LLM生成回复
            code_execution_config=False,  # 禁用代码执行
            human_input_mode="NEVER",  # 不请求人工输入
        )
        return agent

    def create_agent(self, user_name):
        agent = ConversableAgent(
            name=user_name,
            llm_config=self.llm_config,
            system_message=f"""You are an {user_name} expert. Your task is to analyze the given AUTHOR'S TEXT and determine the MBTI personality type of the user based on four binary dimensions: 
1. **Extraversion (E) vs. Introversion (I)** 
2. **Sensing (S) vs. Intuition (N)** 
3. **Thinking (T) vs. Feeling (F)** 
4. **Judging (J) vs. Perceiving (P)** 
For each dimension, provide: 
- **Classification**: Your decision (e.g., "E" or "I"). 
- **Reason**: A brief explanation of why you made this classification. 
- **Confidence level**: A value between 0.0 and 1.0 that reflects how certain you are about the classification. 
Use the following format for your response: 
```
1. Extraversion (E) vs. Introversion (I): [Your decision, e.g., "E"] 
Reason: [Explain the reasoning behind your choice, e.g., "The user frequently discusses external social activities and expresses energy from interactions with others."] 
Confidence level: [Your confidence score, e.g., "0.8"] 
2. Sensing (S) vs. Intuition (N): [Your decision, e.g., "N"] 
Reason: [Explain the reasoning behind your choice, e.g., "The user focuses on abstract ideas and future possibilities rather than concrete details."] 
Confidence level: [Your confidence score, e.g., "0.7"] 
3. Thinking (T) vs. Feeling (F): [Your decision, e.g., "T"] 
Reason: [Explain the reasoning behind your choice, e.g., "The user emphasizes logical analysis and objectivity in decision-making."] 
Confidence level: [Your confidence score, e.g., "0.9"] 
4. Judging (J) vs. Perceiving (P): [Your decision, e.g., "P"] 
Reason: [Explain the reasoning behind your choice, e.g., "The user prefers flexible approaches and is open to last-minute changes."] 
Confidence level: [Your confidence score, e.g., "0.5"] 
``` 
Analyze the AUTHOR'S TEXT carefully, and provide a detailed and thoughtful response for each dimension.""",
            description=f"""{user_name} expert, skilled in analyzing user information from a {
                user_name} angle to predict their MBTI personality type.""",
            human_input_mode="NEVER",
        )
        return agent

    def commentator(self):
        Commentator = ConversableAgent(
            name="Commentator",
            llm_config=self.llm_config,
            system_message="""You are an MBTI personality expert. Please read a piece of text posted by a user, as well as the conclusions given by three experts, and select the most likely MBTI personality type from the conclusions based on the original text. Provide the answer directly without analysis.""",
            description="""Review Expert, to conduct the final analysis and summary.""",
            human_input_mode="NEVER",
        )
        return Commentator

    def first_chats(self, task):
        task = f"""AUTHOR'S TEXT: {task}"""
        first_chats_list = [
            initiate_chats([
                self.chat_unit(
                    self.agent_dict['user_proxy'], first_chat, task),
            ]) for first_chat in [self.agent_dict['Semantic'], self.agent_dict['Sentiment'], self.agent_dict['Linguistic']]
        ]
        temp = []
        for chat in first_chats_list:
            temp.append(chat[0].chat_history[1])
        self.chat_result['first_chats'] = temp
        return first_chats_list

    def circle_chat(self, task, chats, nums, max_depth=3):
        if nums > max_depth:
            return
        # 重复一遍
        chat_prompts = []
        for chat in chats:
            chat_content = chat[0].chat_history[1]
            message = f"""The following are speculations from {chat_content['name']} experts, just for reference, you can stick to your own opinion:
        {chat_content['content']}"""
            chat_prompts.append(message)
        self.chat_result[f'round_{nums}'] = chat_prompts
        combined_prompt = "\n".join(chat_prompts)

        # 初始化下一轮的聊天
        next_chats = [
            initiate_chats([
                self.chat_unit(self.agent_dict['user_proxy'], agent,
                               f"""{task}\n\n{combined_prompt}""")
            ]) for agent in [self.agent_dict['Semantic'], self.agent_dict['Sentiment'], self.agent_dict['Linguistic']]
        ]

        return self.circle_chat(task, next_chats, nums + 1, max_depth)

    # 按照新的框架，在结束讨论后，我们应当进入一个投票环节， 这个环节主要在于讨论
    @staticmethod
    def check_vote(circle_chats: str):
        expert_votes = ['Semantic', 'Sentiment', 'Linguistic']
        for expert in expert_votes:
            txt = f"The following are speculations from {expert} experts, just for reference, you can stick to your own opinion:"
            if expert not in circle_chats:
                continue
            voter = expert
        circle_chats = circle_chats.replace(txt, '')
        mbti_predict = re.findall(r'\d\..*?\[(\S+)\]', circle_chats, re.I)
        Confidence = re.findall(
            r'Confidence level: (\d+\.\d+)', circle_chats, re.I)
        Confidence = [float(i) for i in Confidence]
        Reason = re.findall(r"(Reason\:.*)\n", circle_chats, re.I)
        temp = []
        for i in range(4):
            temp.append([mbti_predict[i], Confidence[i], Reason[i]])
        return voter, temp

    def vote(self):
        circle_chat_final = self.chat_result[f'round_{self.max_round}']
        vote_dict = {}
        for circle_chats in circle_chat_final:
            voter, circle_chat = self.check_vote(circle_chats)
            vote_dict[voter] = circle_chat
        # 记录提取结果
        self.chat_result['vote_dict'] = vote_dict
        # 数据结构 ‘E’: [‘投票数量’， '平均分'， {'投票人':Reason}]
        mbti_vote = {
            'E': [0, 0, {}],
            'I': [0, 0, {}],
            'N': [0, 0, {}],
            'S': [0, 0, {}],
            'T': [0, 0, {}],
            'F': [0, 0, {}],
            'J': [0, 0, {}],
            'P': [0, 0, {}]
        }

        for expert, datas in vote_dict.items():

            for data in datas:
                vote_aim = mbti_vote[data[0]]
                sum_pre = vote_aim[0]*vote_aim[1]
                vote_aim[0] += 1
                sum_pre += data[1]
                vote_aim[1] = sum_pre/vote_aim[0]
                vote_aim[2][expert] = data[2]
        # 记录计算结果
        self.chat_result['mbti_vote'] = mbti_vote

        def vote_result(mbti_vote_final, vote1, vote2):
            if mbti_vote_final[vote1][0] > mbti_vote_final[vote2][0]:
                mbti_vote_final.pop(vote2)
            else:
                mbti_vote_final.pop(vote1)
        mbti_vote_final = mbti_vote.copy()
        vote_result(mbti_vote_final, "E", "I")
        vote_result(mbti_vote_final, "N", "S")
        vote_result(mbti_vote_final, "T", "F")
        vote_result(mbti_vote_final, "J", "P")
        self.chat_result['mbti_vote_final'] = mbti_vote_final
        self.chat_result['vote_predict'] = "".join(mbti_vote_final.keys())

    def final_predict(self, nums, task):
        final_predict = f"""### Original text of the user's statement.
{task}\n\n
### Experts' conclusions
{self.chat_result[f'round_{nums}']}""".replace(', just for reference, you can stick to your own opinion:', ':')
        final_predict = initiate_chats([
            self.chat_unit(
                self.agent_dict['user_proxy'], self.agent_dict['Commentator'], final_predict)
        ])
        agent_result = final_predict[0].chat_history[1]
        self.chat_result['commentator_response'] = agent_result

    def result_clean(self):
        print(self.chat_result['commentator_response'])
        mbti_type = re.findall(r"[E|I][S|N][T|F][J|P]",
                               self.chat_result['commentator_response']['content'])
        if len(mbti_type) > 1:
            mbti_type = mbti_type[0]
        self.chat_result['final_predict'] = mbti_type[0]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    @log_to_file
    def run(self, task):
        task = data_process(task, deepclean=self.deepclean, cutoff=self.cutoff)
        self.chat_result['origin_task'] = task
        first_chats = self.first_chats(task)
        self.circle_chat(task, first_chats, 1, self.max_round)
        self.vote()
        # self.final_predict(self.max_round, task)
        # self.result_clean()
        return self.chat_result
