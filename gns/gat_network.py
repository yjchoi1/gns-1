from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from graph_network import build_mlp
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            nnode_in_features: int,
            nnode_out_features: int,
            nmlp_layers: int,
            mlp_hidden_dim: int):

        super(Encoder, self).__init__()
        # Encode node features as an MLP
        self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                                 [mlp_hidden_dim
                                                  for _ in range(nmlp_layers)],
                                                 nnode_out_features),
                                       nn.LayerNorm(nnode_out_features)])

    def forward(self, x: torch.tensor):
        return self.node_fn(x)


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            heads):

        super(GraphAttentionNetwork, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(
            GATConv(in_channels, hidden_channels, heads=heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        # Output layer
        self.layers.append(
            GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index)

        # Output layer
        x = self.layers[-1](x, edge_index)

        return x


class Decoder(nn.Module):
    """The Decoder: :math: `\mathcal{G} \rightarrow \mathcal{Y}` extracts the
    dynamics information from the nodes of the final latent graph,
    :math: `y_i = \delta v (v_i^M)`

    """

    def __init__(
            self,
            nnode_in: int,
            nnode_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int):
        """The Decoder coder's learned function, :math: `\detla v`, is an MLP.
        After the Decoder, the future position and velocity are updated using an
        Euler integrator, so the :math: `yi` corresponds to accelerations,
        :math: `\"{p}_i`, with 2D or 3D dimension, depending on the physical domain.

        Args:
          nnode_in: Number of node inputs (latent dimension of size 128).
          nnode_out: Number of node outputs (particle dimension).
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
        """
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(
            nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

    def forward(self,
                x: torch.tensor):
        """The forward hook runs when the Decoder class is instantiated

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in)

        """
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    def __init__(
            self,
            nnode_in_features: int,
            nnode_out_features: int,
            latent_dim: int,
            in_gat_channels: int,
            hidden_gat_channels: int,
            out_gat_channels: int,
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
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = GraphAttentionNetwork(
            in_channels=in_gat_channels,
            hidden_channels=hidden_gat_channels,
            out_channels=out_gat_channels,
            num_layers=nmessage_passing_steps,
            heads=attention_heads
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor):
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
        x = self._encoder(x)
        x = self._processor(x, edge_index)
        x = self._decoder(x)
        return x


# interaction_network // graph_attention_network
