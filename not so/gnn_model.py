import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            adjacency_matrix: Adjacency matrix tensor of shape (num_nodes, num_nodes)

        Returns:
            logits: Output tensor of shape (num_nodes, output_dim)
        """
        # Message passing: aggregate neighbors features
        agg = torch.matmul(adjacency_matrix, x)  # simple aggregation by adjacency

        # First layer
        h = F.relu(self.fc1(agg))

        # Second layer / output
        logits = self.fc2(h)

        return logits
