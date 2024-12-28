import pathlib
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from collections import deque, defaultdict
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

def process_single_node(args):
    # Previous process_single_node implementation remains the same
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

    def save(self, path: str):
        """Save the dataset to disk"""
        save_dict = {
            'data_list': self.data_list,
            'csv_paths': self.csv_paths,
            'k_hops': self.k_hops,
            'split': self.split,
            'train_ratio': self.train_ratio
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str):
        """Load a dataset from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        dataset = cls(
            csv_paths=save_dict['csv_paths'],
            k_hops=save_dict['k_hops'],
            split=save_dict['split'],
            train_ratio=save_dict['train_ratio']
        )
        dataset.data_list = save_dict['data_list']
        return dataset

    def _process_csv_file(self, csv_path: str) -> List[Data]:
        # Previous _process_csv_file implementation remains the same
        graphs = []
        df = pd.read_csv(csv_path)
        all_serials = df['serial'].values
        n_nodes = len(all_serials)
        max_serial = df['serial'].max()

        n_samples = min(2000, n_nodes)
        sampled_indices = np.random.choice(n_nodes, n_samples, replace=False)
        sampled_serials = all_serials[sampled_indices]

        n_train = int(n_samples * self.train_ratio)
        shuffled_indices = np.random.permutation(n_samples)

        if self.split == 'train':
            selected_serials = sampled_serials[shuffled_indices[:n_train]]
        else:
            selected_serials = sampled_serials[shuffled_indices[n_train:]]

        process_args = [
            (csv_path, int(serial), self.k_hops, max_serial)
            for serial in selected_serials
        ]

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

def create_and_save_datasets(
    csv_paths: List[str],
    output_dir: Path,
    k_hops: int = 2,
    train_ratio: float = 0.8,
    num_workers: int = None,
    seed: Optional[int] = None
):
    """Create and save the datasets"""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = NodeParallelSearchGraphDataset(
        csv_paths, k_hops, 'train', train_ratio, seed, num_workers
    )
    test_dataset = NodeParallelSearchGraphDataset(
        csv_paths, k_hops, 'test', train_ratio, seed, num_workers
    )

    # Save datasets
    train_dataset.save(output_dir / 'train_dataset.pkl')
    test_dataset.save(output_dir / 'test_dataset.pkl')

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Saved datasets to {output_dir}")

def load_and_create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Load datasets and create DataLoader objects"""
    # Load datasets
    train_dataset = NodeParallelSearchGraphDataset.load(dataset_dir / 'train_dataset.pkl')
    test_dataset = NodeParallelSearchGraphDataset.load(dataset_dir / 'test_dataset.pkl')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.parent
    csv_dir = root_dir / 'csv_output_mini'
    output_dir = root_dir / 'datasets'

    csv_paths = list(csv_dir.glob('*.csv'))
    print(f"Found {len(csv_paths)} CSV files")

    # Use 80% of available CPU cores by default
    num_workers = max(1, int(mp.cpu_count() * 0.8))
    print(f"Using {num_workers} workers per CSV file")

    # Create and save datasets
    create_and_save_datasets(
        csv_paths,
        output_dir,
        k_hops=2,
        train_ratio=0.8,
        num_workers=num_workers,
        seed=42
    )

    # Load datasets and create dataloaders
    train_loader, test_loader = load_and_create_dataloaders(
        output_dir,
        batch_size=32
    )

    print("Successfully created DataLoaders")
    # You can now use train_loader and test_loader for training