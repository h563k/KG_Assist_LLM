import os
import yaml
import pandas as pd


class ModelConfig:
    def __init__(self) -> None:
        self.file_path = os.path.abspath(__file__)
        self.file_path = os.path.dirname(os.path.dirname(self.file_path))
        self.configs = self.config_read()
        self.OpenAI = self.configs['OpenAI']
        self.mbti = self.configs['MBTI']

    def config_read(self):
        file_path = os.path.join(self.file_path, 'config/setting.yaml')
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @property
    def save_path(self):
        return self.configs['LIWC']['path']

    @property
    def cats_id(self):
        return self.configs['LIWC']['cats_id']

    def mbti_data(self, dataset):
        assert dataset in ['kaggle', 'pand'], 'dataset must be kaggle or pand'
        data_path = self.mbti[dataset]
        data_path = os.path.join(self.file_path, data_path)
        data = pd.read_csv(data_path)
        return data
