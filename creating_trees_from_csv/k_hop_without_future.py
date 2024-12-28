import pathlib
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from collections import deque, defaultdict
import numpy as np
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

def get_csv_paths(csv_dir) -> list:
    return list(csv_dir.glob('*.csv'))

def process_single_node(args):
    csv_path, center_node_serial, k_hops, max_serial = args
    try:
        df = pd.read_csv(csv_path)
        edges = defaultdict(set)
        for _, row in df.iterrows():
            if pd.notna(row['father_serial']) and row['serial'] <= center_node_serial and row['father_serial'] <= center_node_serial:
                edges[int(row['father_serial'])].add(int(row['serial']))

        nodes_to_keep = set()
        queue = deque([(center_node_serial, 0)])
        visited = {center_node_serial}

        while queue:
            node, depth = queue.popleft()
            nodes_to_keep.add(node)

            if depth < k_hops:
                children = edges.get(node, set())
                parent_row = df[df['serial'] == node]
                parent = parent_row['father_serial'].iloc[0] if not parent_row.empty else None
                if parent == 0 or pd.isna(parent):
                    parent = None

                neighbors = list(children)
                if parent is not None and parent <= center_node_serial:
                    neighbors.append(int(parent))

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        node_mapping = {}
        node_features = []
        sorted_nodes = sorted(nodes_to_keep)

        for idx, serial in enumerate(sorted_nodes):
            node_mapping[serial] = idx
            node_data = df[df['serial'] == serial].iloc[0]
            features = [
                node_data['f'], node_data['h'], node_data['g'], node_data['BF'],
                node_data['h0'], node_data['H_min'], node_data['last_H_min_update'],
                node_data['f_max']
            ]
            node_features.append(features)

        edge_list = []
        for node in nodes_to_keep:
            children = edges.get(node, set())
            parent_row = df[df['serial'] == node]
            parent = parent_row['father_serial'].iloc[0] if not parent_row.empty else None
            if parent == 0 or pd.isna(parent):
                parent = None

            for child in children:
                if child in nodes_to_keep:
                    edge_list.extend([
                        [node_mapping[node], node_mapping[child]],
                        [node_mapping[child], node_mapping[node]]
                    ])

            if parent is not None and parent in nodes_to_keep:
                parent = int(parent)
                edge_list.extend([
                    [node_mapping[node], node_mapping[parent]],
                    [node_mapping[parent], node_mapping[node]]
                ])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        center_idx = node_mapping[center_node_serial]
        y = torch.tensor(center_node_serial / max_serial, dtype=torch.float)

        return {
            'x': x,
            'edge_index': edge_index,
            'center_idx': center_idx,
            'y': y
        }
    except Exception as e:
        print(f"Error processing node {center_node_serial}: {str(e)}")
        return None

class NodeParallelSearchGraphDataset(Dataset):
    def __init__(self, csv_paths: List[str], k_hops: int = 2, split: str = 'train',
                 train_ratio: float = 0.8, seed: Optional[int] = None, num_workers: int = None):
        super().__init__()
        self.csv_paths = csv_paths
        self.k_hops = k_hops
        self.split = split
        self.train_ratio = train_ratio
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.8))

        if seed is not None:
            np.random.seed(seed)

        self.data_list = self._process_files()

    def _process_csv_file(self, csv_path: str) -> List[Data]:
        graphs = []
        df = pd.read_csv(csv_path)
        all_serials = df['serial'].values
        n_nodes = len(all_serials)
        max_serial = df['serial'].max()

        # Sample nodes
        n_samples = min(2000, n_nodes)
        sampled_indices = np.random.choice(n_nodes, n_samples, replace=False)
        sampled_serials = all_serials[sampled_indices]

        # Split into train/test
        n_train = int(n_samples * self.train_ratio)
        shuffled_indices = np.random.permutation(n_samples)

        if self.split == 'train':
            selected_serials = sampled_serials[shuffled_indices[:n_train]]
        else:
            selected_serials = sampled_serials[shuffled_indices[n_train:]]

        # Prepare arguments for parallel processing
        process_args = [
            (csv_path, int(serial), self.k_hops, max_serial)
            for serial in selected_serials
        ]

        # Process nodes in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=len(selected_serials),
                     desc=f'Processing nodes from {Path(csv_path).name}',
                     leave=False) as pbar:
                for result in executor.map(process_single_node, process_args):
                    if result is not None:
                        graph = Data(
                            x=result['x'],
                            edge_index=result['edge_index'],
                            center_idx=result['center_idx'],
                            y=result['y']
                        )
                        graphs.append(graph)
                    pbar.update(1)

        return graphs

    def _process_files(self):
        all_data = []

        # Process one CSV file at a time
        with tqdm(total=len(self.csv_paths), desc=f'Processing {self.split} set CSV files') as pbar:
            for csv_path in self.csv_paths:
                graphs = self._process_csv_file(csv_path)
                all_data.extend(graphs)
                pbar.update(1)

        return all_data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def create_node_parallel_dataloaders(csv_paths: List[str], k_hops: int = 2, batch_size: int = 32,
                                   train_ratio: float = 0.8, num_workers: int = None, seed: Optional[int] = None):
    train_dataset = NodeParallelSearchGraphDataset(
        csv_paths, k_hops, 'train', train_ratio, seed, num_workers
    )
    test_dataset = NodeParallelSearchGraphDataset(
        csv_paths, k_hops, 'test', train_ratio, seed, num_workers
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader

if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.parent
    output_dir = root_dir / 'csv_output'
    csv_paths = get_csv_paths(output_dir)
    print(f"Found {len(csv_paths)} CSV files")

    # Use 80% of available CPU cores by default
    num_workers = max(1, int(mp.cpu_count() * 0.8))
    print(f"Using {num_workers} workers per CSV file")

    train_loader, test_loader = create_node_parallel_dataloaders(
        csv_paths,
        k_hops=2,
        batch_size=32,
        train_ratio=0.8,
        num_workers=num_workers,
        seed=42
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Save dataloaders
    with open(output_dir / 'train_loader_node_parallel.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open(output_dir / 'test_loader_node_parallel.pkl', 'wb') as f:
        pickle.dump(test_loader, f)
    print("Saved dataloaders to output directory")