from itertools import product
from random import randint, uniform

import torch
from rich.progress import track
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


class StraightLineDataLoader(DataLoader):
    """
    Subclass of the PyTorch Geometric Dataloader, a dataset of 2D straight lines.

    All generated lines lie along the x axis. They have a (uniform) random number of
    nodes and a (uniform) random length - see num_nodes and line_length args.

    Parameters
    ----------
    num_data_examples : int
        Number of straight lines to generate.
    min_num_nodes : int, optional
        Minimum number of nodes in each line, by default 4
    max_num_nodes : int, optional
        Maximum number of nodes in each line, by default 10
    min_line_length : float, optional
        Minimum total length of the line, by default 5.0
    max_line_length : float, optional
        Maximum total length of the line, by default 10.0
    batch_size : int, optional
        Batch size of the returned DataLoader, by default 64
    verbose : bool, optional
        If False, progress bar will not be shown while generating lines, by default True
    """

    def __init__(
        self,
        num_data_examples: int,
        min_num_nodes: int = 4,
        max_num_nodes: int = 10,
        min_line_length: float = 5.0,
        max_line_length: float = 10.0,
        batch_size: int = 64,
        verbose: bool = True,
    ):

        self.num_data_examples = num_data_examples

        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes

        self.min_line_length = min_line_length
        self.max_line_length = max_line_length

        self.verbose = verbose

        self.straight_lines: list[Data] = self.compute_straight_lines()

        super().__init__(self.straight_lines, batch_size=batch_size)

    def compute_straight_lines(self) -> list[Data]:
        """
        Generate a list of straight lines, oriented along the x axis.

        Returns
        -------
        list[Data]
            List of straight lines.
        """
        straight_lines = []
        for _ in track(range(self.num_data_examples), disable=not self.verbose):
            num_nodes = randint(self.min_num_nodes, self.max_num_nodes)
            line_length = uniform(self.min_line_length, self.max_line_length)

            node_x_coordinates = torch.linspace(
                -line_length, line_length, num_nodes, dtype=torch.float32
            )
            x = torch.stack(
                [
                    node_x_coordinates,
                    torch.zeros_like(node_x_coordinates),
                ]
            ).T

            fully_connected_edge_indices = torch.tensor(
                [
                    (u, v)
                    for u, v in product(
                        torch.arange(0, num_nodes), torch.arange(0, num_nodes)
                    )
                    if u != v
                ]
            ).T

            # As a precaution - should already be fully connected.
            edge_index = to_undirected(fully_connected_edge_indices)

            data = Data(x=x, edge_index=edge_index)
            straight_lines.append(data)

        return straight_lines
