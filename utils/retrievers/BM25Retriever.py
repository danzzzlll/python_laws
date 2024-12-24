from typing import List
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
import joblib
from tqdm import tqdm

from config import settings


class BMRetriever:
    def __init__(self, settings):
        if settings.bm_index:
            self.nodes = joblib.load(settings.bm_index)

    def load_bm_retriever(self):
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=settings.top_k_bm,
            stemmer=Stemmer.Stemmer("russian"),
            language="russian",
        )
        return bm25_retriever
    
    def make_nodes(self, docs: Document):
        nodes = []
        for doc in tqdm(docs):
            node = TextNode(text=doc.get_content())
            node.metadata = doc.metadata
            nodes.append(node)

        joblib.dump(nodes, settings.bm_index)
        print(f"Saved bm nodes to {settings.bm_index}")
