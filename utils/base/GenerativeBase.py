import os
from abc import ABC, abstractmethod

class GenerativeBaseModel(ABC):
    def __init__(self,
                 model_name,
                 system_prompt=None):
        
        self.model_name = model_name
        self.system_prompt = system_prompt

        
    @abstractmethod
    def inference(self,
                  text, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def config_prompt(self):
        pass