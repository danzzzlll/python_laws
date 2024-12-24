import os
from abc import ABC, abstractmethod

from utils.base.GenerativeBase import GenerativeBaseModel
from langchain.chat_models.gigachat import GigaChat

from config import settings


class GigaApi(GenerativeBaseModel):
    def __init__(self,
                 model_name=settings.model_name,
                 credentials=settings.api_key,):
        
        super().__init__(model_name)
        self.credentials = credentials
        self.chat = None
        self.load()

    def config_prompt(self):
        return super().config_prompt()

    def load(self):
        """Load the model and tokenizer."""
        self.chat = GigaChat(model="GigaChat:latest", credentials=self.credentials, verify_ssl_certs=False)
        
    def inference(self,
                  prompt):
        """Generate text based on the provided input"""
        res = self.chat.invoke(prompt)
        
        return res.content