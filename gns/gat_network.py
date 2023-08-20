from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GAT
from graph_network import build_mlp
from graph_network import Encoder
from graph_network import Decoder
import torch.nn.functional as F


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            heads,
            use_layernorm=True):
        super(GraphAttentionNetwork, self).__init__()

        self.gat_layers = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            norm=torch.nn.LayerNorm(hidden_channels) if use_layernorm is True else None,
            heads=heads,
            concat=True
        )

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            self.gat_layers(x, edge_index, edge_attr=edge_attr)
        else:
            self.gat_layers(x, edge_index)
        return x


class EncodeProcessDecode(nn.Module):
    def __init__(
            self,
            nnode_in_features: int,
            nnode_out_features: int,
            nedge_in_features: int,
            latent_dim: int,
            in_gat_channels: int,
            hidden_gat_channels: int,
            nmessage_passing_steps: int,
            attention_heads: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
    ):
        """Encode-Process-Decode function approximator for learnable simulator.

        Args:
          nnode_in_features: Number of node input features (for 2D = 30,
            calculated as [10 = 5 times steps * 2 positions (x, y) +
            4 distances to boundaries (top/bottom/left/right) +
            16 particle type embeddings]).
          nnode_out_features:  Number of node outputs (particle dimension).
          nedge_in_features: Number of edge input features (for 2D = 3,
            calculated as [2 (x, y) relative displacements between 2 particles +
            distance between 2 particles]).
          latent_dim: Size of latent dimension (128)
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

        """
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = GraphAttentionNetwork(
            in_channels=in_gat_channels,
            hidden_channels=hidden_gat_channels,
            out_channels=latent_dim,
            num_layers=nmessage_passing_steps,
            heads=attention_heads,
            use_layernorm=True
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor,
                edge_features: torch.tensor):

        """The forward hook runs at instatiation of EncodeProcessorDecode class.

          Args:
            x: Particle state representation as a torch tensor with shape
              (nparticles, nnode_in_features)
            edge_index: A torch tensor list of source and target nodes with shape
              (2, nedges)

          Returns:
            x: Particle state representation as a torch tensor with shape
              (nparticles, nnode_out_features)
        """
        x, edge_features = self._encoder(x, edge_features)
        x = self._processor(x, edge_index, edge_features)
        x = self._decoder(x)
        return x

# interaction_network // graph_attention_network
