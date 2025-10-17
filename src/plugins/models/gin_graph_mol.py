"""Graph Multi-Domain Molecular GIN model plugin with complete model implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from src.plugins.base import BaseModelPlugin, InputType, ModelMetadata, ModelType

logger = logging.getLogger(__name__)


# ===== Model Architecture Implementation =====


class MLP(nn.Module):
    """Multi-layer perceptron for GIN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_bn = use_bn

        # First layer
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        if use_bn:
            self.norms.append(nn.BatchNorm1d(output_dim))

    def forward(self, x):
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            if self.use_bn:
                x = self.norms[i](x)
            x = F.relu(x)
        x = self.linears[-1](x)
        if self.use_bn:
            x = self.norms[-1](x)
        return x


class GNN(nn.Module):
    """GIN (Graph Isomorphism Network) backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.5,
        use_bn: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = MLP(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2,
                use_bn=use_bn
            )
            self.convs.append(GINConv(nn=mlp, train_eps=True))

    def forward(self, x, edge_index, batch):
        """Forward pass through GNN layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)
        return x


class GINGraphClassifier(nn.Module):
    """GIN model for graph classification with atom encoder."""

    def __init__(
        self,
        num_classes: int,
        num_atom_features: int = 200,
        hidden_dim: int = 64,
        num_gnn_layers: int = 5,
        dropout: float = 0.5,
        use_bn: bool = True,
        num_node_features: int = 7,  # Default for multi-domain molecular datasets
    ):
        super().__init__()

        # Atom feature encoder (for categorical features)
        self.encoder_atom = nn.ModuleList([
            nn.Embedding(num_atom_features, hidden_dim)
        ])

        # Encoder for continuous features (single feature)
        self.encoder_single = nn.Linear(1, hidden_dim)

        # Encoder for multi-dimensional continuous features
        self.encoder_multi = nn.Linear(num_node_features, hidden_dim)

        # GNN backbone
        self.gnn = GNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            use_bn=use_bn
        )

        # Linear projection after pooling
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Final classifier
        self.clf = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with attributes:
                - x: node features [num_nodes, num_features]
                - edge_index: edge connectivity [2, num_edges]
                - batch: batch assignment [num_nodes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode node features
        # Try atom embedding first (for categorical features)
        if x.dtype == torch.long and x.dim() == 1:
            x = self.encoder_atom[0](x)
        elif x.dtype == torch.long and x.dim() == 2 and x.shape[1] == 1:
            x = self.encoder_atom[0](x.squeeze(-1))
        else:
            # For continuous features
            if x.dim() == 1:
                x = x.unsqueeze(-1)

            # Check if single feature or multi-feature
            if x.shape[1] == 1:
                x = self.encoder_single(x)
            else:
                x = self.encoder_multi(x)

        # GNN forward pass
        x = self.gnn(x, edge_index, batch)

        # Linear projection
        x = self.linear(x)

        # Classification
        logits = self.clf(x)

        return logits


# ===== Plugin Implementation =====


class GINGraphMolPlugin(BaseModelPlugin):
    """GIN model plugin for Graph Multi-Domain Molecular dataset - trained with FedSAK."""

    # ===== Metadata definition =====
    metadata = ModelMetadata(
        model_id="gin_graph_mol_fedsak",
        name="GIN-GraphMol-FedSAK",
        model_type=ModelType.CONVOLUTIONAL_NEURAL_NETWORK,  # GNN is a type of CNN
        input_type=InputType.GRAPH,
        description="Graph Isomorphism Network trained on Multi-Domain Molecular dataset with FedSAK",
        numeric_id=2,
        author="Federated Learning Team",
        version="1.0.0",
        tags=["federated", "graph", "gnn", "gin", "molecular"]
    )

    # ===== Configuration definition =====
    dataset_plugin_id = "graph_multi_domain_mol"
    model_path = "GIN_fedsak"
    checkpoint_pattern = "client/client_model_{client_id}.pt"

    # Model hyperparameters
    model_config = {
        "num_atom_features": 200,
        "num_node_features": 7,  # Default for TUDataset molecular graphs
        "hidden_dim": 64,
        "num_gnn_layers": 5,
        "dropout": 0.5,
        "use_bn": True
    }

    def build_model(self, num_labels: int, **kwargs) -> nn.Module:
        """Build model instance."""
        num_atom_features = kwargs.get("num_atom_features", self.model_config["num_atom_features"])
        num_node_features = kwargs.get("num_node_features", self.model_config["num_node_features"])
        hidden_dim = kwargs.get("hidden_dim", self.model_config["hidden_dim"])
        num_gnn_layers = kwargs.get("num_gnn_layers", self.model_config["num_gnn_layers"])
        dropout = kwargs.get("dropout", self.model_config["dropout"])
        use_bn = kwargs.get("use_bn", self.model_config["use_bn"])

        logger.info(f"Building {self.metadata.name} model:")
        logger.info(f"  Num classes: {num_labels}")
        logger.info(f"  Num atom features: {num_atom_features}")
        logger.info(f"  Num node features: {num_node_features}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  GNN layers: {num_gnn_layers}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Use BatchNorm: {use_bn}")

        model = GINGraphClassifier(
            num_classes=num_labels,
            num_atom_features=num_atom_features,
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_bn=use_bn
        )

        return model

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(state, dict) and "model" in state:
            logger.debug("Extracting model state from checkpoint dict")
            model_state = state["model"]
            if "cur_round" in state:
                logger.info(f"  Checkpoint from round: {state['cur_round']}")
            return model_state

        return state

    def infer_hyperparams_from_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Infer hyperparameters from state dict."""
        # Infer hidden_dim from encoder (try multiple encoder names)
        if "encoder_multi.weight" in state_dict:
            hidden_dim = state_dict["encoder_multi.weight"].shape[0]
            num_node_features = state_dict["encoder_multi.weight"].shape[1]
        elif "encoder_single.weight" in state_dict:
            hidden_dim = state_dict["encoder_single.weight"].shape[0]
            num_node_features = 1
        elif "encoder.weight" in state_dict:
            hidden_dim = state_dict["encoder.weight"].shape[0]
            num_node_features = state_dict["encoder.weight"].shape[1]
        else:
            hidden_dim = 64
            num_node_features = 7

        # Infer num_gnn_layers by counting conv layers
        num_gnn_layers = len([k for k in state_dict.keys() if k.startswith("gnn.convs.") and k.endswith(".nn.linears.0.weight")])
        if num_gnn_layers == 0:
            num_gnn_layers = self.model_config["num_gnn_layers"]

        # Infer num_atom_features from embedding
        if "encoder_atom.0.weight" in state_dict:
            num_atom_features = state_dict["encoder_atom.0.weight"].shape[0]
        elif "encoder_atom.atom_embedding_list.0.weight" in state_dict:
            num_atom_features = state_dict["encoder_atom.atom_embedding_list.0.weight"].shape[0]
        else:
            num_atom_features = self.model_config["num_atom_features"]

        logger.debug(f"Inferred hyperparameters from state dict:")
        logger.debug(f"  hidden_dim: {hidden_dim}")
        logger.debug(f"  num_gnn_layers: {num_gnn_layers}")
        logger.debug(f"  num_atom_features: {num_atom_features}")
        logger.debug(f"  num_node_features: {num_node_features}")

        return {
            "hidden_dim": hidden_dim,
            "num_gnn_layers": num_gnn_layers,
            "num_atom_features": num_atom_features,
            "num_node_features": num_node_features,
            "dropout": self.model_config["dropout"],
            "use_bn": self.model_config["use_bn"]
        }
