"""Research knowledge graph built on NetworkX.

Extracts entities (Papers, Methods, Datasets, Metrics) and relations
(CITES, EXTENDS, OUTPERFORMS) from literature and experiment results,
enabling research gap discovery and trend analysis.
"""

from research.knowledge.graph.entities import Entity, EntityType
from research.knowledge.graph.relations import Relation, RelationType
from research.knowledge.graph.builder import KnowledgeGraphBuilder
from research.knowledge.graph.query import KnowledgeGraphQuery

__all__ = [
    "Entity",
    "EntityType",
    "Relation",
    "RelationType",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphQuery",
]
