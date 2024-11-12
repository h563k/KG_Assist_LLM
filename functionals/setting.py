import yaml
import argparse
import pandas as pd

class ModelConfig:
    def __init__(self) -> None:
        self.args = self.args_read()
        self.configs = self.config_read()
        self.liwc = self.configs['LIWC']
        self.llm = self.configs['LLM']
        self.ollama = self.configs['OLLAMA']

    @staticmethod
    def config_read():
        with open('config/setting.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def args_read():
        parser = argparse.ArgumentParser(description='这是一个示例程序。')
        # 添加参数
        parser.add_argument('--dashscope_key', type=bool)
        # 解析命令行参数
        args = parser.parse_args()
        return args

    @property
    def save_path(self):
        return self.configs['LIWC']['path']

    @property
    def cats_id(self):
        return self.configs['LIWC']['cats_id']

    @property
    def mbti_data(self):
        data_path = self.configs['MBTI']['data_path']
        print(data_path)
        data = pd.read_csv(data_path)
        return data