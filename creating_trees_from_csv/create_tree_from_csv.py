import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Load CSV file
csv_path = ROOT_DIR / "try.csv"
data_df = pd.read_csv(csv_path)

# Extract relevant columns
node_ids = data_df["level_1#"].values  # Node IDs
father_ids = data_df["level_2#"].values  # Father IDs
features_df = data_df.drop(columns=["level_1#", "level_2#"])  # Node features

# Convert node features to PyTorch tensor
x = torch.tensor(features_df.values, dtype=torch.float)

# Build edges based on father-child relationships
edge_index = []
for child_id, father_id in zip(node_ids, father_ids):
    if not pd.isna(father_id):  # Ensure father_id is valid
        edge_index.append([int(father_id), int(child_id)])

# Convert edges to PyTorch tensor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create the PyTorch Geometric Data object
graph = Data(x=x, edge_index=edge_index)

print(graph)
