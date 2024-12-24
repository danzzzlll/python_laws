## config

from pydantic import Field
from typing import ClassVar, List
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv


class Settings(BaseSettings):
    #gigachat
    model_name: str = Field("GigaChat:latest")
    api_key: str = Field(...)
    #local_model
    model_path:str = r"C:\Мисис\models\rubert_turbo\models--sergeyzh--rubert-tiny-turbo\snapshots\93769a3baad2b037e5c2e4312fccf6bcfe082bf1"
    # cache_folder: ClassVar[str] = "../../models/rubert_turbo"
    # device: ClassVar[str] = "cpu"
    #retrievers
    bm_index: str = Field("indexes/bm_index/nodes.pkl")
    emb_index: str = Field("indexes/emb_index/")
    top_k_bm: int = Field(20)
    top_k_emb: int = Field(20)
    # fusion
    weights: List[float] = Field([0.4, 0.6])
    fusion_top_k: int = 10


    class Config:
        env_file = ".env"


settings = Settings()