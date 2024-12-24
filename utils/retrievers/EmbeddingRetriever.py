import pandas as pd
import faiss
from llama_index.core import Document
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import settings
from tqdm import tqdm


class EmbeddingRetriever:
    def __init__(self, settings):
        # self.data = pd.read_csv(settings.emb_index)
        self.model = HuggingFaceEmbedding(settings.model_path)


    def make_docs(self):
        docs = []
        for ind, row in tqdm(self.data.iterrows(), total=len(self.data)):
            doc = Document(text=row['chunk'])
            doc.metadata = {
                "law_number": row['title'],
                "title": row['new_number'],
                "page_first": row['page_first'],
                "page_last": row['page_last'] 
            }
            docs.append(doc)
        return docs
    

    def make_and_save_index(self):
        d = 312
        faiss_index = faiss.IndexFlatL2(d)
        docs = self.make_docs()

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, show_progress=True, embed_model=self.model
        )

        index.storage_context.persist(settings.emb_index)

        print(f'index Save to {settings.emb_index}')

    
    def load_emb_index(self):
        vector_store = FaissVectorStore.from_persist_dir(settings.emb_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=settings.emb_index
        )
        index = load_index_from_storage(storage_context=storage_context, embed_model=self.model)
        return index