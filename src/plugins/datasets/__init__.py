"""Dataset plugins."""

from src.plugins.datasets.domainnet_plugin import DomainNetPlugin
from src.plugins.datasets.graph_multi_domain_mol_plugin import GraphMultiDomainMolPlugin

__all__ = ['DomainNetPlugin', 'GraphMultiDomainMolPlugin']
