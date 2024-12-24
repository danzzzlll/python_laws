from llama_index.core.schema import NodeWithScore
from typing import List, Dict
from collections import defaultdict

from utils.dto.FusionDTO import FusionDTO
from utils.retrievers.BM25Retriever import BMRetriever
from utils.retrievers.EmbeddingRetriever import EmbeddingRetriever
from config import settings

class FusionRetriever(BMRetriever, EmbeddingRetriever):
    def __init__(self, settings):
        BMRetriever.__init__(self, settings=settings)
        EmbeddingRetriever.__init__(self, settings=settings)

        self.bm25_r = self.load_bm_retriever()
        emb_index = self.load_emb_index()
        self.emb_r = emb_index.as_retriever(similarity_top_k=settings.top_k_emb)
    

    def normalize_scores(self, nodes: List[NodeWithScore]) -> List[FusionDTO]:
        """
        Преобразует список NodeWithScore в список FusionDTO и нормализует поле score по методу мин-макс.

        Аргументы:
            nodes (List[NodeWithScore]): Список объектов NodeWithScore для обработки.

        Возвращает:
            List[FusionDTO]: Список объектов FusionDTO с нормализованными значениями score.
        """
        dtos = [FusionDTO.from_node_with_score(node) for node in nodes]
        scores = [dto.score for dto in dtos]

        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            for dto in dtos:
                dto.score = 1.0  # Все значения одинаковы, нормализуем в 1.0
            return dtos

        for dto in dtos:
            dto.score = (dto.score - min_score) / (max_score - min_score)

        return dtos
    

    def rrf_fusion(self, list1: List[FusionDTO], list2: List[FusionDTO], k: int = 60, weights = settings.weights) -> List[FusionDTO]:
        """
        Выполняет RRF-фузию (Reciprocal Rank Fusion) для двух списков объектов FusionDTO, 
        учитывая ранги и веса скоров.

        Аргументы:
            list1 (List[FusionDTO]): Первый список объектов FusionDTO.
            list2 (List[FusionDTO]): Второй список объектов FusionDTO.
            k (int): Параметр RRF для контроля влияния ранга (по умолчанию: 60).
            weight1 (float): Вес первого списка (по умолчанию: 1.0).
            weight2 (float): Вес второго списка (по умолчанию: 1.0).

        Возвращает:
            List[FusionDTO]: Список объектов FusionDTO, объединённых по методу RRF, 
                            отсортированный по убыванию финальных RRF-оценок.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        dto_mapping: Dict[str, FusionDTO] = {}

        def update_rrf_scores(dtos: List[FusionDTO], rank_offset: int, weight: float):
            """
            Обновляет RRF-оценки для объектов списка с учётом их текущего score.

            Аргументы:
                dtos (List[FusionDTO]): Список FusionDTO для обработки.
                rank_offset (int): Значение k, добавляемое к рангу.
                weight (float): Вес текущего списка.
            """
            for rank, dto in enumerate(dtos, start=1):
                identifier = f"{dto.law_number}-{dto.title}"
                adjusted_score = dto.score * weight
                rrf_scores[identifier] += adjusted_score / (rank + rank_offset)
                dto_mapping[identifier] = dto 

        update_rrf_scores(list1, k, weights[0])
        update_rrf_scores(list2, k, weights[1])

        combined = [
            (dto_mapping[identifier], score) for identifier, score in rrf_scores.items()
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in combined]

        
    def retrieve(self, query):
        nodes_bm = self.bm25_r.retrieve(query)
        nodes_emb = self.emb_r.retrieve(query)

        dtos_bm = self.normalize_scores(nodes_bm)
        dtos_emb = self.normalize_scores(nodes_emb)

        return self.rrf_fusion(dtos_bm, dtos_emb)[:settings.fusion_top_k]