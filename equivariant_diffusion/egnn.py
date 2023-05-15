from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Tensor
from torch_geometric.utils import scatter, to_dense_batch


class EquivariantGCL(MessagePassing):
    r"""
    Equivariant Graph Convolutional Layer.

    Largely based off:
    Hoogeboom, Emiel, Victor Garcia Satorras, Clément Vignac, and Max Welling.
    "Equivariant Diffusion for Molecule Generation in 3D". arXiv, 16 June 2022.
    http://arxiv.org/abs/2203.17003.
    (See Appendix B in particular).

    Parameters
    ----------
    message_dimension : int
        Dimension of the message vectors, $\vec{m}_{ij}$ in the paper.
    embedding_input_dimension : int
        Dimension of the input node features, $\vec{h}^l$ in Eq. 12.
    embedding_output_dimension : int
        Dimension of the output node features, $\vec{h}^{l+1}$ in Eq. 12.
    edge_attribute_dimension : int
        Dimension of the edge attribute, $a_{ij}$ in Eq. 12.
    """

    def __init__(
        self,
        message_dimension: int,
        embedding_input_dimension: int,
        embedding_output_dimension: int,
        edge_attribute_dimension: int,
        **kwargs,
    ):
        super(EquivariantGCL, self).__init__(aggr="sum", **kwargs)

        self.message_embedding_network = nn.Sequential(
            nn.Linear(
                2 * embedding_input_dimension + 1 + edge_attribute_dimension,
                embedding_output_dimension,
            ),
            nn.SiLU(),
            nn.Linear(embedding_output_dimension, message_dimension),
            nn.SiLU(),
        )
        # TODO: Check if this is needed? Included as it is used by Hoogeboom et al.
        self.message_inference_network = nn.Sequential(
            nn.Linear(message_dimension, 1),
            nn.Sigmoid(),
        )
        # Note Hoogeboom et al. model has hl added to the output here, which requires
        # dimensions are equal. However, my starting dimension is 1, so I will skip
        # this step.
        self.feature_embedding_network = nn.Sequential(
            nn.Linear(
                embedding_input_dimension + message_dimension,
                embedding_output_dimension,
            ),
            nn.SiLU(),
            nn.Linear(embedding_output_dimension, embedding_output_dimension),
            nn.SiLU(),
        )
        self.direction_weighting_network = nn.Sequential(
            nn.Linear(
                2 * embedding_input_dimension + 1 + edge_attribute_dimension,
                embedding_output_dimension,
            ),
            nn.SiLU(),
            nn.Linear(embedding_output_dimension, embedding_output_dimension),
            nn.SiLU(),
            nn.Linear(embedding_output_dimension, 1),
        )

    def forward(
        self,
        x: Tensor,
        features: Tensor,
        edge_index: Adj,
        edge_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Equivariant GCL.

        Computes updated position (x) and feature vectors, such that the returned
        feature vectors are invariant to translations and rotations of input positions,
        and the returned positions are equivariant to translations and rotations of the
        input positions.

        Parameters
        ----------
        x : Tensor
            Input position tensor.
        features : Tensor
            Input "node features" tensor.
        edge_index : Adj
            Edge indices.
        edge_features : Tensor
            Edge feature tensor.

        Returns
        -------
        Tuple[Tensor, Tensor]
            First tuple entry is the updated position tensor, second is the updated
            feature tensor.
        """
        relative_positions = x[edge_index[0]] - x[edge_index[1]]
        relative_distances = (relative_positions**2).sum(dim=-1, keepdim=True).sqrt()

        sum_m_ij = self.propagate(
            edge_index,
            x=x,
            features=features,
            relative_distances=relative_distances,
            edge_features=edge_features,
        )
        new_features = self.feature_embedding_network(
            torch.cat(
                [features, sum_m_ij],
                dim=-1,
            )
        )

        # See Eq. 12, Hoogeboom et al.
        direction_vectors = relative_positions / (relative_distances + 1.0)
        direction_weights = self.direction_weighting_network(
            torch.cat(
                [
                    features[edge_index[0]],
                    features[edge_index[1]],
                    relative_distances,
                    edge_features,
                ],
                dim=-1,
            )
        )
        x_updates = direction_vectors * direction_weights
        x_updates = scatter(
            x_updates,
            edge_index[0],
            dim=0,
            dim_size=x.size(0),
            reduce="sum",
        )
        new_x = x + x_updates

        return new_x, new_features

    def message(
        self,
        features_i: Tensor,
        features_j: Tensor,
        relative_distances: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """
        Compute the "message embeddings" using PyTorch Geometric message passing.

        Parameters
        ----------
        features_i : Tensor
            Features of the ith nodes (in the edge index tensor).
        features_j : Tensor
            Features of the jth nodes (in the edge index tensor).
        relative_distances : Tensor
            Relative distances between ith and jth nodes.
        edge_features : Tensor
            Edge feature vectors.

        Returns
        -------
        Tensor
            Non-aggregated messages.
        """
        m_ij = self.message_embedding_network(
            torch.cat(
                [
                    features_i,
                    features_j,
                    relative_distances,
                    edge_features,
                ],
                dim=-1,
            )
        )
        e_ij_tilde = self.message_inference_network(m_ij)
        return e_ij_tilde * m_ij


class EquivariantGNN(nn.Module):
    """
    Equivariant Graph Neural Network, composed of Equivariant GCLs.

    As above, largely based off:
    Hoogeboom, Emiel, Victor Garcia Satorras, Clément Vignac, and Max Welling.
    "Equivariant Diffusion for Molecule Generation in 3D". arXiv, 16 June 2022.
    http://arxiv.org/abs/2203.17003.
    (See Appendix B in particular).

    Parameters
    ----------
    hidden_embedding_dimension : int
        Dimension to use for the node features in the "hidden" GCLs.
    message_dimension : int
        Dimension of the messages passed.
    num_layers : int
        Number of GCLs.
    num_timesteps : int
        Number of timesteps of the diffusion process. Used to "normalise" time tensor
        t to [0.0, 1.0].
    """

    def __init__(
        self,
        hidden_embedding_dimension: int,
        message_dimension: int,
        num_layers: int,
        num_timesteps: int,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        layers = []
        for i in range(num_layers):
            layer = EquivariantGCL(
                message_dimension,
                1 if i == 0 else hidden_embedding_dimension,
                hidden_embedding_dimension,
                1,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: Tensor, edge_index: Adj, t: Tensor, batch_indices: Tensor
    ) -> Tensor:
        """
        Forward pass of the Equivariant GNN.

        Parameters
        ----------
        x : Tensor
            Initial node position tensor.
        edge_index : Adj
            Edge indices.
        t : Tensor
            Timestep tensor, one per node. Note this will not necessarily be the same
            value across the entire tensor since x may contain different batches - see
            https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
        batch_indices : Tensor
            Batch index of each node.

        Returns
        -------
        Tensor
            "Velocity" for each node; the difference between its "final position" output
            by the GCLs and its initial position. Note this is "centred" to give it
            mean 0.
        """
        intial_position = x
        initial_relative_positions = x[edge_index[0]] - x[edge_index[1]]
        initial_relative_distances = (
            (initial_relative_positions**2).sum(dim=-1, keepdim=True).sqrt()
        )

        features = (t / self.num_timesteps).unsqueeze(dim=1)
        for layer in self.layers:
            x, features = layer(x, features, edge_index, initial_relative_distances)

        vel = x - intial_position
        dense_batch_vel, _ = to_dense_batch(vel, batch=batch_indices, fill_value=0.0)
        dense_vel_CoMs = dense_batch_vel.mean(axis=1)
        vel -= dense_vel_CoMs[batch_indices]

        return vel
