"""Model plugins."""

from src.plugins.models.vit_domainnet import DomainNetViTPlugin
from src.plugins.models.gin_graph_mol import GINGraphMolPlugin

__all__ = ['DomainNetViTPlugin', 'GINGraphMolPlugin']
