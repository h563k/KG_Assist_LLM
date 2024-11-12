from autogen import AssistantAgent, UserProxyAgent
from functionals.setting import ModelConfig


class MBTIAgent:
    def __init__(self, agent_name) -> None:
        self.config = ModelConfig()

    def env(self):
        config_list = [
            {
                "model": self.config.ollama['model_name'],
                "base_url": f"{self.config.ollama['base_url']}/v1",
                "api_key": "ollama",
            }
        ]
        self.agent = AssistantAgent(
            name="assistant", llm_config={"config_list": config_list})
        
        