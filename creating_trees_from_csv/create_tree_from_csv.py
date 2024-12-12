import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def create_graph_from_csv(csv_path):
    """
    Creates a PyTorch Geometric graph object from a CSV file.

    Parameters:
        csv_path (str or Path): Path to the CSV file.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    # Load CSV file
    data_df = pd.read_csv(csv_path)

    # Extract relevant columns
    node_ids = data_df["serial"].values  # Node IDs
    father_ids = data_df["father_serial"].values  # Father IDs
    node_max = data_df["node_max"].values  # Target feature
    features_df = data_df.drop(columns=["serial", "father_serial", "node_max"])  # Node features

    # Convert node features to PyTorch tensor
    x = torch.tensor(features_df.values, dtype=torch.float)

    # Convert target feature to PyTorch tensor
    y = torch.tensor(node_max, dtype=torch.float)

    # Build edges based on father-child relationships
    edge_index = []
    for child_id, father_id in zip(node_ids, father_ids):
        if not pd.isna(father_id):  # Ensure father_id is valid
            edge_index.append([int(father_id), int(child_id)])

    # Convert edges to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create and return the PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index, y=y)


def plot_tree_graph(graph):
    """
    Plots a tree-like graph using NetworkX from a PyTorch Geometric graph object.

    Parameters:
        graph (Data): PyTorch Geometric Data object.
    """
    # Convert to NetworkX graph
    G = to_networkx(graph, to_undirected=False)  # Keep it directed to find hierarchy

    # Define a hierarchical layout function for a tree
    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        If the graph is a tree, this will return the positions to plot it in a hierarchical layout.
        """
        if not nx.is_tree(G):
            raise TypeError("The graph is not a tree and hierarchy_pos is designed for tree layouts.")

        # If no root is specified, choose a node with no incoming edges (a possible root)
        if root is None:
            # This works only if there's one root. If multiple candidates exist, choose one.
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if len(roots) == 0:
                raise ValueError("No root found. The graph might not be a tree or all nodes have incoming edges.")
            root = roots[0]

        def _hierarchy_pos(G, root, left, right, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = (right - left)/len(children)
                nextx = left + dx/2
                for child in children:
                    pos = _hierarchy_pos(G, child, left=nextx - dx/2, right=nextx + dx/2,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
                    nextx += dx
            return pos

        return _hierarchy_pos(G, root, 0, width, vert_loc=vert_loc, xcenter=xcenter)

    # Get hierarchical positions
    pos = hierarchy_pos(G)

    # Plot the graph as a tree
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Tree Graph")
    # Save the plot
    plt.savefig('tree_graph.png')
    plt.show()

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent
    csv_path = ROOT_DIR / "try.csv"

    # Create graph from CSV
    graph = create_graph_from_csv(csv_path)

    # Plot the tree-like graph
    plot_tree_graph(graph)
    