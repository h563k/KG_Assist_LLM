import os
import yaml
import pandas as pd


class ModelConfig:
    def __init__(self) -> None:
        self.file_path = os.path.abspath(__file__)
        self.file_path = os.path.dirname(os.path.dirname(self.file_path))
        self.configs = self.config_read()
        self.liwc = self.configs['LIWC']
        self.llm = self.configs['LLM']
        self.ollama = self.configs['OLLAMA']
        self.OpenAI = self.configs['OpenAI']


    @staticmethod
    def config_read():
        with open('config/setting.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config

    @property
    def save_path(self):
        return self.configs['LIWC']['path']

    @property
    def cats_id(self):
        return self.configs['LIWC']['cats_id']

    @property
    def mbti_data(self):
        data_path = self.configs['MBTI']['data_path']
        data_path = os.path.join(self.file_path, data_path)
        data = pd.read_csv(data_path)
        return data
