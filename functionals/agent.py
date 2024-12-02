import re
from autogen import initiate_chats, ConversableAgent
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file


class MbtiChats:
    def __init__(self, task, max_round=3, nums='five', openai_type='openai_hk') -> None:
        self.nums = nums
        self.task = self.data_process(task)
        self.llm_config = self.env_init(openai_type)
        self.chat_result = {'origin_task': task}
        self.agent_dict = {
            "user_proxy": self.user_proxy(),
            "Semantic": self.create_agent("Semantic"),
            "Sentiment": self.create_agent("Sentiment"),
            "Linguistic": self.create_agent("Linguistic"),
        }
        self.save_result()

    def env_init(self, openai_type) -> None:
        config = ModelConfig()
        openai_config = config.OpenAI['openai_origin']
        openai_config = config.OpenAI[openai_type]
        config_list = [
            {
                "model": "gpt-3.5-turbo",
                "base_url": openai_config['base_url'],
                "api_type": "openai",
                "api_key": openai_config['api_key'],
                "temperature": 0.2,
            }
        ]
        llm_config = {"config_list": config_list}
        return llm_config

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

    @staticmethod
    def data_process(txt: str):
        temp = []
        txt = txt.split('|||')
        for message in txt:
            website = re.findall('(https://\S+|http://\S+)', message)
            if not website:
                temp.append(message)
                continue
            for web in website:
                message = message.replace(web, '')
            temp.append(message) if message else None
        txt = "\n".join(temp)
        return txt

    def create_agent(self, user_name):
        agent = ConversableAgent(
            name=user_name,
            llm_config=self.llm_config,
            system_message=f"""Please analyze and predict the users' MBTI personality types from a {user_name} perspective, with the number of predictions being no more than {self.nums}.""",
            description=f"""{user_name} expert, skilled in analyzing user information from a {user_name} angle to predict their MBTI personality type.""",
            human_input_mode="NEVER",
        )
        return agent

    def user_proxy(self):
        agent = ConversableAgent(
            name="MessageForwarderAgent",
            system_message="You are a message forwarder, and your task is to forward the received messages unaltered to the next recipient.",
            llm_config=False,  # 不使用LLM生成回复
            code_execution_config=False,  # 禁用代码执行
            human_input_mode="NEVER",  # 不请求人工输入
        )
        return agent

    def first_chats(self):
        first_chats_list = [
            initiate_chats([
                self.chat_unit(
                    self.agent_dict['user_proxy'], first_chat, self.task),
            ]) for first_chat in [self.agent_dict['Semantic'], self.agent_dict['Sentiment'], self.agent_dict['Linguistic']]
        ]
        temp = []
        for chat in first_chats_list:
            temp.append(chat[0].chat_history[1])
        self.chat_result['first_chats'] = temp
        return first_chats_list

    def circle_chat(self, chats, nums, max_depth=3):
        if nums >= max_depth:
            return
        # 重复一遍
        chat_prompts = []
        for chat in chats:
            chat_content = chat[0].chat_history[1]
            message = f"""The following are speculations from {chat_content['name']} experts, just for reference, you can stick to your own opinion:
        {chat_content['content']}"""
            chat_prompts.append(message)

        combined_prompt = "\n".join(chat_prompts)
        self.chat_result[f'round_{nums}'] = combined_prompt

        # 初始化下一轮的聊天
        next_chats = [
            initiate_chats([
                self.chat_unit(self.agent_dict['user_proxy'], agent,
                               f"""{self.task}\n\n{combined_prompt}""")
            ]) for agent in [self.agent_dict['Semantic'], self.agent_dict['Sentiment'], self.agent_dict['Linguistic']]
        ]

        return self.circle_chat(next_chats, nums + 1, max_depth)

    @log_to_file
    def save_result(self):
        return self.chat_result
