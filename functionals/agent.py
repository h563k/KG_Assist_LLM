import os
import re
import time
from typing import List, Dict
from autogen import initiate_chats, ConversableAgent
from functionals.system_config import ModelConfig
from functionals.standard_log import log_to_file, debug
from functionals.data_clean import data_process
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

config = ModelConfig()
mbti = config.mbti


class MbtiChats:
    def __init__(self, max_round=mbti['max_round'], openai_type=mbti['openai_type'], deepclean=mbti['deepclean'], cutoff=mbti['cutoff']) -> None:
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
        self.openai_type = openai_type
        self.model = config.OpenAI[openai_type]['model']
        self.llm_config = self.env_init(openai_type)
        self.max_round = max_round
        self.chat_result = {}
        self.agent_dict = {
            "user_proxy": self.user_proxy(),
            "Semantic": self.create_agent("Semantic"),
            "Sentiment": self.create_agent("Sentiment"),
            "Linguistic": self.create_agent("Linguistic"),
            "Commentator": self.commentator(),
            "Single": self.create_agent('personality analysis')
        }
        self.deepclean = deepclean
        self.cutoff = cutoff
        self.turn = {"H": "Y", "L": "N"}

    def env_init(self, openai_type) -> Dict:
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
        if openai_type == 'ollama':
            config_list[0]['price'] = [0, 0]
        cache_seed = config.OpenAI['cache_seed']
        llm_config = {"config_list": config_list,
                      "cache_seed": None if not cache_seed else cache_seed,
                      }
        return llm_config

    @staticmethod
    def chat_unit(sender, recipient, message):
        return {
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "summary_method": "last_msg",
            "max_turns": 1,
            "clear_history": False
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
    def create_agent_cot(self, user_name):
        agent = ConversableAgent(
            name="COT",
            llm_config=self.llm_config,
            system_message=f"""```
You are {user_name}, a specialist OCEAN analyst focusing exclusively on semantic/sentiment/linguistic patterns. Examine the AUTHOR'S TEXT through your disciplinary lens using this thinking chain:

1. **Text Analysis** (COT Step 1):
- Semantic Expert: "Identify key themes, abstract concepts, and subject-object relationships..."
- Sentiment Expert: "Detect emotional valence, subjective evaluations, and affect-loaded expressions..."
- Linguistic Expert: "Analyze syntactic patterns, discourse markers, and lexical preferences..."

2. **Trait Indicators** (COT Step 2):
- Semantic Expert: "...focus on content substance over delivery style"
- Sentiment Expert: "...prioritize emotional resonance in decision-making cues"
- Linguistic Expert: "...examine structural features like pronoun frequency and tense usage"

3. **Dimension Classification** (COT Step 3):
Apply your specialized knowledge to each OCEAN axis (H for high, L for low):

[O] Openness to experience:
- Classification: [Your decision, e.g., "H"]
- Confidence Level: ___
- Reasoning Chain: "The text shows ___ → which suggests ___ → therefore..."

[C] Conscientiousness:
- Classification: [Your decision, e.g., "H"]
- Confidence Level: ___
- Reasoning Chain: "Patterns of ___ → indicate ___ → leading to..."

[E] Extraversion:
- Classification: [Your decision, e.g., "H"]
- Confidence Level: ___
- Reasoning Chain: "Language features like ___ → indicate ___ → leading to..."

[A] Agreeableness:
- Classification: [Your decision, e.g., "L"]
- Confidence Level: ___
- Reasoning Chain: "Patterns of ___ → demonstrate ___ → resulting in..."

[N] Neuroticism:
- Classification: [Your decision, e.g., "L"]
- Confidence Level: ___
- Reasoning Chain: "Structural elements including ___ → imply ___ → concluding..."

**Final Output Format:**
1. Openness to experience:
Classification: ["H"] or ["L"]

2. Conscientiousness:
Classification: ["H"] or ["L"]


3. Extraversion:
Classification: ["H"] or ["L"]


4. Agreeableness: 
Classification: ["H"] or ["L"]

5. Neuroticism:
Classification: ["H"] or ["L"]
```
""",
            description="A helpful assistant that helps users with their questions.",
            human_input_mode="NEVER",
        )
        return agent

    def create_agent(self, user_name):
        agent = ConversableAgent(
            name=user_name,
            llm_config=self.llm_config,
            system_message=f"""You are an {user_name} expert. Your task is to analyze the given AUTHOR'S TEXT and determine the OCEAN personality type of the user based on four binary dimensions:
1. **Openness to experience**
2. **Conscientiousness**
3. **Extraversion**
4. **Agreeableness**
5. **Neuroticism**
For each dimension, provide:
- **Classification**: Your decision (e.g., "H" or "L"). 
- **Reason**: A brief explanation of why you made this classification.
- **Confidence level**: A value between 0.0 and 1.0 that reflects how certain you are about the classification.
Use the following format for your response (H means high, L means low):
```
1. Openness to experience:
Classification: [e.g., "H"]
Reason: [Explain the reasoning behind your choice briefly]
Confidence level: [e.g., "0.8"]

2. Conscientiousness:
Classification: [e.g., "L"]
Reason: [Explain the reasoning behind your choice briefly]
Confidence level: [e.g., "0.7"]

3. Extraversion:
Classification: [e.g., "H"]
Reason: [Explain the reasoning behind your choice briefly]
Confidence level: [e.g., "0.9"]

4. Agreeableness: 
Classification: [e.g., "L"]
Reason: [Explain the reasoning behind your choice briefly]
Confidence level: [e.g., "0.5"]

5. Neuroticism: 
Classification: [e.g., "H"]
Reason: [Explain the reasoning behind your choice briefly]
Confidence level: [e.g., "0.6"]
```""",
            description=f"""{user_name} expert, skilled in analyzing user information from a {
                user_name} angle to predict their MBTI personality type.""",
            human_input_mode="NEVER",
        )
        return agent

    def commentator(self):
        Commentator = ConversableAgent(
            name="Commentator",
            llm_config=self.llm_config,
            system_message="""You are an arbiter with expertise in the OCEAN domain. Please read the given AUTHOR'S TEXT and carefully review the following solutions from Semantic, Sentiment, and Linguistic agents as additional information, determine the OCEAN personality type .

Use the following format for your response (H means high, L means low):
```
1. **Classification**: [Your decision, e.g., "H" or "L"] 
2. **Reason**: [Explain the reasoning behind your choice briefly"]
3. **Confidence level**: [Your confidence score, e.g., "0.8"] """,
            description="""Review Expert, to conduct the final analysis and summary.""",
            human_input_mode="NEVER",
        )
        return Commentator

    def first_chats(self, task):
        temp = []
        task = f"""AUTHOR'S TEXT: {task}"""
        first_chats_list = []
        for first_chat in [self.agent_dict['Semantic'], self.agent_dict['Sentiment'], self.agent_dict['Linguistic']]:
            first_chats_list.append(initiate_chats([
                self.chat_unit(
                    self.agent_dict['user_proxy'], first_chat, task),
            ]))
            if self.openai_type == "ollama":
                time.sleep(0.1)
        for chat in first_chats_list:
            temp.append(chat[0].chat_history[1])
        self.chat_result['first_chats'] = temp

        return first_chats_list

    @staticmethod
    def score_reset(score: float):
        if score == 1.0:
            return 1.0
        elif 0.9 <= score < 1.0:
            return 0.8
        elif 0.8 <= score < 0.9:
            return 0.5
        elif 0.6 <= score < 0.8:
            return 0.3
        else:
            return 0.1

    def circle_chat(self, task, chats, nums, max_depth):
        if nums > max_depth:
            return
        if nums > 1:
            print("start self check")
            circle_chats = self.chat_result[f'round_{nums-1}']
            temps = []
            for circle_chat in circle_chats:
                _, temp = self.check_vote(circle_chat)
                temps.append(temp)
            voter_lists = set()
            for voter_list in temps:
                voter_result = ""
                for voter in voter_list:
                    voter_result += voter[0]
                voter_lists.add(voter_result)
            if len(voter_lists) == 1:
                print("end self circle chat")
                self.chat_result[f'round_{max_depth}'] = circle_chats
                return
        # 重复一遍
        chat_prompts = {}  # 用于交给专家判断
        chat_results = []  # 记录原始结果
        for chat in chats:
            chat_content = chat[0].chat_history[1]
            message = f"""The following are speculations from {chat_content['name']} experts, just for reference, you can stick to your own opinion:
        {chat_content['content']}"""
            chat_prompts[chat_content['name']] = message
            chat_results.append(message)
        self.chat_result[f'round_{nums}'] = chat_results
        # 初始化下一轮的聊天
        name_list = ['Semantic', 'Sentiment', 'Linguistic']
        next_chats = []
        for name in name_list:
            temp = []
            for expert in name_list:
                if expert != name:
                    temp.append(chat_prompts[expert])
            combined_prompt = "\n".join(temp)
            agent = self.agent_dict[name]
            next_chats.append(initiate_chats([
                self.chat_unit(self.agent_dict['user_proxy'], agent,
                               f"""{combined_prompt}""")
            ]))
            if self.openai_type == "ollama":
                time.sleep(0.1)
        return self.circle_chat(task, next_chats, nums + 1, max_depth)

    @staticmethod
    def get_mbti_predict(circle_chats: str) -> List:
        if "Final Output" in circle_chats:
            circle_chats = circle_chats.split("Final Output")[-1]
        mbti_predict = re.findall(
            r'Classification.*?(H|L)', circle_chats)
        if mbti_predict:
            print(circle_chats)
            print(mbti_predict)
            return mbti_predict
        else:
            return []

    # 按照新的框架，在结束讨论后，我们应当进入一个投票环节， 交给法官角色做最后判断
    def check_vote(self, circle_chats: str):
        expert_votes = ['Semantic', 'Sentiment', 'Linguistic']
        voter = None
        for expert in expert_votes:
            txt = f"The following are speculations from {expert} experts, just for reference, you can stick to your own opinion:\n "
            if txt in circle_chats:
                voter = expert
                break
        circle_chats = circle_chats.replace(txt, '').strip()
        print('step4-2')
        mbti_predict = self.get_mbti_predict(circle_chats)
        Confidence = re.findall(
            r'\n.*?Confidence.*?(\d+\.\d+)', circle_chats, re.I)
        Confidence = [float(i) for i in Confidence]
        Reason = re.findall(r"Reason(.*)\n",
                            circle_chats, re.I)
        temp = []
        for i in range(5):
            result = [mbti_predict[i], Confidence[i], Reason[i]]
            temp.append(result)
        print('step4-3')
        print('voter')
        print(voter, temp)
        return voter, temp

    def vote(self):
        circle_chat_final = self.chat_result[f'round_{self.max_round}']
        vote_dict = {}
        print('step4-1')
        for circle_chats in circle_chat_final:
            voter, circle_chat = self.check_vote(circle_chats)
            vote_dict[voter] = circle_chat
            print('step4-3-1')
            print(vote_dict)
        # 记录提取结果
        print('step4-3-2')
        self.chat_result['vote_dict'] = vote_dict
        # 数据结构 ‘E’: [‘投票数量’， '平均分'， {'投票人':Reason}]
        print('step4-3-3')
        mbti_vote = [[0, 0, {}]] * 10
        print('step4-4')
        mbti_vote = [[0, 0, {}] for _ in range(10)]
        for expert, datas in vote_dict.items():
            for i, data in enumerate(datas):
                if data[0] == "H":
                    mbti_vote[2*i][0] += 1
                    mbti_vote[2*i][1] = max(mbti_vote[2*i][1], data[1])
                    mbti_vote[2*i][2][expert] = data[2]
                elif data[0] == "L":
                    mbti_vote[2*i+1][0] += 1
                    mbti_vote[2*i+1][1] = max(mbti_vote[2*i+1][1], data[1])
                    mbti_vote[2*i+1][2][expert] = data[2]
        print(mbti_vote)
        # 记录计算结果
        for vote_aim in mbti_vote:
            vote_aim[1] = self.score_reset(vote_aim[1])
        print('step4-5')
        print(mbti_vote)
        self.chat_result['mbti_vote'] = mbti_vote

    def battle(self, ocean_num, types, task):
        mbti_vote = self.chat_result['mbti_vote']
        vote1_data = mbti_vote[ocean_num*2]
        vote2_data = mbti_vote[ocean_num*2 + 1]
        # print('vote1_data', vote1_data)
        # print('vote2_data', vote2_data)
        # 2位专家均给出0.5以上分数不进入辩论环节
        # 投票
        if vote1_data[0] > vote2_data[0]:
            mbti_type = "H"
            if vote1_data[1] > 0.5:
                self.chat_result['final_mbti'].append(mbti_type)
                return
            elif vote1_data[0] == 3 and vote1_data[1] < 0.5:
                self.chat_result['final_mbti'].append("L")
                return
        else:
            mbti_type = "L"
            if vote2_data[1] > 0.5:
                self.chat_result['final_mbti'].append(mbti_type)
                return
            elif vote2_data[0] == 3 and vote2_data[1]<0.5:
                self.chat_result['final_mbti'].append("H")
                return
        print('step5-1')
        vote1_reason = "\nReason ".join(
            vote1_data[2].values()) if vote1_data[2] else ""
        vote2_reason = "\nReason ".join(
            vote2_data[2].values()) if vote2_data[2] else ""
        vote1_content = f"""\nthere are {" ".join(vote1_data[2].keys())} agents think the **Classification** is H.
    the **Reason** is:
    Reason {vote1_reason}.
    the **Confidence level** is {vote1_data[1]}."""
        vote2_content = f"""\nthere are {" ".join(vote2_data[2].keys())} agents think the **Classification** is L.
    the **Reason** is:ß
    Reason {vote2_reason}.
    the **Confidence level** is {vote2_data[1]}."""
        battle_content = f"""### AUTHOR'S TEXT
    {task}\n\n
    ### Experts' solutions
    In the OCEAN dimension of type **{types}**"""
        if vote1_data[2].keys():
            battle_content += vote1_content
        if vote2_data[2].keys():
            battle_content += vote2_content
        print(battle_content)
        self.chat_result[f'battle_content_{types}'] = battle_content
        final_predict = initiate_chats([
            self.chat_unit(
                self.agent_dict['user_proxy'], self.agent_dict['Commentator'], battle_content)
        ])
        if self.openai_type == "ollama":
            time.sleep(0.1)
        agent_result = final_predict[0].chat_history[1]['content']
        self.chat_result[f'battle_{types}'] = agent_result
        predict = self.get_mbti_predict(agent_result)
        print('step5-2')
        print(predict)
        self.chat_result['final_mbti'].append("".join(predict))

    def final_predict(self, task):
        self.chat_result['final_mbti'] = []
        mbti_types = ["Openness to experience", "Conscientiousness",
                      "Extraversion", "Agreeableness", "Neuroticism"]
        for i, mbti_type in enumerate(mbti_types):
            self.battle(i, mbti_type, task)
        self.chat_result['final_mbti'] = "".join(
            self.chat_result['final_mbti'])
        self.chat_result['final_mbti'] = self.chat_result['final_mbti'].replace(
            "H", "Y").replace("L", "N")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
    @log_to_file
    def run(self, task):
        print('step1')
        task = data_process(task, cutoff=self.cutoff)
        self.chat_result['origin_task'] = task
        print('step2')
        first_chats = self.first_chats(task)
        print('step3')
        self.circle_chat(task, first_chats, 1, self.max_round)
        print('step4')
        self.vote()
        print('step5')
        self.final_predict(task)
        print('step6')
        return self.chat_result

    # 消融4
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
    @log_to_file
    def run_single(self, task):
        task = data_process(task, cutoff=self.cutoff)
        self.chat_result['origin_task'] = task
        task = f"""AUTHOR'S TEXT: {task}"""
        chat = initiate_chats([
            self.chat_unit(
                self.agent_dict['user_proxy'], self.agent_dict['Single'], task),
        ])
        circle_chats = chat[0].chat_history[1]['content']
        print("start get mbti predict cot")
        temp = self.get_mbti_predict(circle_chats)
        temp = [self.turn[x] for x in temp]
        print(temp, "final_mbti")
        self.chat_result['final_mbti'] = "".join(temp)

    # 消融6
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
    @log_to_file
    def run_without_vote(self, task):
        task = data_process(task, cutoff=self.cutoff)
        self.chat_result['origin_task'] = task
        first_chats = self.first_chats(task)
        self.circle_chat(task, first_chats, 1, self.max_round)
        ExpertsSolutions = self.chat_result['round_3']
        ExpertsSolutions = "\n\n".join(ExpertsSolutions)
        task = f"""AUTHOR'S TEXT: {task}\n\n### Experts' solutions\n\n{ExpertsSolutions}"""
        chats = initiate_chats([
            self.chat_unit(
                self.agent_dict['user_proxy'], self.agent_dict['Single'], task),
        ])
        chats = chats[0].chat_history[1]['content']
        chats = chats.split("\n\n")
        final_mbti = self.get_mbti_predict(chats)
        self.chat_result['final_mbti'] = "".join(final_mbti)
        return self.chat_result

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
    @log_to_file
    def run_cot(self, task):
        print("cot_version")
        self.agent_dict['Single'] = self.create_agent_cot(
            'personality analysis')
        self.run_single(task)


class MbtiTwoAgent(MbtiChats):
    def __init__(self, max_round=mbti['max_round'], openai_type=mbti['openai_type'], deepclean=mbti['deepclean'], cutoff=mbti['cutoff']) -> None:
        super().__init__(max_round, openai_type, deepclean, cutoff)
        self.agent_list = ["Semantic", "Sentiment", "Linguistic"]
        self.delet_agent = ["Linguistic"]
        self.agent_dict.pop(self.delet_agent[0])
        self.agent_list = [self.agent_dict[x]
                           for x in self.agent_list if x not in self.delet_agent]

    def first_chats(self, task):
        temp = []
        task = f"""AUTHOR'S TEXT: {task}"""
        first_chats_list = []
        for first_chat in self.agent_list:
            first_chats_list.append(initiate_chats([
                self.chat_unit(
                    self.agent_dict['user_proxy'], first_chat, task),
            ]))
        for chat in first_chats_list:
            temp.append(chat[0].chat_history[1])
        self.chat_result['first_chats'] = temp
        return first_chats_list

    def check_vote(self, circle_chats: str):
        expert_votes = ["Semantic", "Sentiment", "Linguistic"]
        print('expert_votes')
        print(expert_votes)
        voter = None
        for expert in expert_votes:
            txt = f"The following are speculations from {expert} experts, just for reference, you can stick to your own opinion:\n "
            if txt in circle_chats:
                voter = expert
                break
        circle_chats = circle_chats.replace(txt, '').strip()
        print('step4-2')
        print(circle_chats)
        mbti_predict = self.get_mbti_predict(circle_chats)
        Confidence = re.findall(
            r'\n.*?Confidence.*?(\d+\.\d+)', circle_chats, re.I)
        Confidence = [float(i) for i in Confidence]
        Reason = re.findall(r"Reason(.*)\n",
                            circle_chats, re.I)
        temp = []
        for i in range(4):
            result = [mbti_predict[i], Confidence[i], Reason[i]]
            temp.append(result)
        print('step4-3')
        print('voter')
        print(voter, temp)
        return voter, temp

    def vote(self):
        circle_chat_final = self.chat_result[f'round_{self.max_round}']
        vote_dict = {}
        print('step4-1')
        for circle_chats in circle_chat_final:
            voter, circle_chat = self.check_vote(circle_chats)
            vote_dict[voter] = circle_chat
            print('step4-3-1')
            print(vote_dict)
        # 记录提取结果
        print('step4-3-2')
        self.chat_result['vote_dict'] = vote_dict
        # 数据结构 ‘E’: [‘投票数量’， '平均分'， {'投票人':Reason}]
        print('step4-3-3')
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
        print('step4-4')
        print(vote_dict)
        for expert, datas in vote_dict.items():
            for data in datas:
                vote_aim = mbti_vote[data[0]]
                vote_aim[0] += 1
                vote_aim[1] = max(vote_aim[1], data[1])
                vote_aim[2][expert] = data[2]
        print(mbti_vote)
        # 记录计算结果
        for _, vote_aim in mbti_vote.items():
            vote_aim[1] = self.score_reset(vote_aim[1])
        print('step4-5')
        print(mbti_vote)
        self.chat_result['mbti_vote'] = mbti_vote

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
        next_chats = []
        for agent in self.agent_list:
            next_chats.append(initiate_chats([
                self.chat_unit(self.agent_dict['user_proxy'], agent,
                               f"""{combined_prompt}""")
            ]))
        return self.circle_chat(task, next_chats, nums + 1, max_depth)

    def battle(self, vote1, vote2, task):
        mbti_vote = self.chat_result['mbti_vote']
        vote1_data = mbti_vote[vote1]
        vote2_data = mbti_vote[vote2]

        if vote1_data[0] == 2:
            mbti_type = vote1
            self.chat_result['final_mbti'].append(mbti_type)
        elif vote2_data[0] == 2:
            mbti_type = vote2
            self.chat_result['final_mbti'].append(mbti_type)
        else:
            vote1_reason = "\n    Reason ".join(vote1_data[2].values())
            vote2_reason = "\n    Reason ".join(vote2_data[2].values())
            vote1_content = f"""\n- type ({vote1})
there are {" ".join(vote1_data[2].keys())} agents think the **Classification** is {vote1}.
the **Reason** is:
    Reason {vote1_reason}.
the **Confidence level** is {vote1_data[1]}."""
            vote2_content = f"""\n- type ({vote2})
there are {" ".join(vote2_data[2].keys())} agents think the **Classification** is {vote2}.
the **Reason** is:
    Reason {vote2_reason}.
the **Confidence level** is {vote2_data[1]}."""
            battle_content = f"""### AUTHOR'S TEXT
    {task}\n\n
### Experts' solutions
In the MBTI dimension of type ({vote1}) vs. type ({vote2}):"""
            if vote1_data[2].keys():
                battle_content += vote1_content
            if vote2_data[2].keys():
                battle_content += vote2_content
            self.chat_result[f'battle_content_{vote1}{vote2}'] = battle_content
            final_predict = initiate_chats([
                self.chat_unit(
                    self.agent_dict['user_proxy'], self.agent_dict['Commentator'], battle_content)
            ])
            if self.openai_type == "ollama":
                time.sleep(0.1)
            agent_result = final_predict[0].chat_history[1]['content']
            self.chat_result[f'battle_{vote1}{vote2}'] = agent_result
            predict = self.get_mbti_predict(agent_result)
            print('step5-2')
            print(predict)
            self.chat_result['final_mbti'].append("".join(predict))

    def final_predict(self, task):
        self.chat_result['final_mbti'] = []
        mbti_types = [['E', 'I'], ['N', 'S'], ['T', 'F'], ['J', 'P']]
        for vote1, vote2 in mbti_types:
            self.battle(vote1, vote2, task)
        self.chat_result['final_mbti'] = "".join(
            self.chat_result['final_mbti'])

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(0))
    @log_to_file
    def run(self, task):
        print('step1-two')
        task = data_process(task, cutoff=self.cutoff)
        self.chat_result['origin_task'] = task
        print('step2-two')
        first_chats = self.first_chats(task)
        print('step3-two')
        self.circle_chat(task, first_chats, 1, self.max_round)
        print('step4-two')
        self.vote()
        print('step5-two')
        self.final_predict(task)
        print('step6-two')
        return self.chat_result
