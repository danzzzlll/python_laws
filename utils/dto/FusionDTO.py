from pydantic import BaseModel
from llama_index.core.schema import NodeWithScore

class FusionDTO(BaseModel):
    score: float
    law_number: str
    title: str
    page_first: int
    page_last: int
    text: str

    @classmethod
    def from_node_with_score(cls, node: NodeWithScore) -> "FusionDTO":
        """
        Create a FusionDTO instance from a NodeWithScore instance.

        Args:
            node (NodeWithScore): The source node object.

        Returns:
            FusionDTO: A new instance of FusionDTO.
        """
        return cls(
            score=node.score,
            law_number=node.metadata.get("law_number", ""),
            title=node.metadata.get("title", ""),
            page_first=node.metadata.get("page_first", 0),
            page_last=node.metadata.get("page_last", 0),
            text=node.text,
        )