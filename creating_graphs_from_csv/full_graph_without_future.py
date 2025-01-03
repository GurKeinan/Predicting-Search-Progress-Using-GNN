import torch
from torch_geometric.data import Data, Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import pickle

MAX_SERIAL = 20000

class EfficientSearchGraphDataset(Dataset):
    def __init__(self,
                 csv_paths: List[str],
                 split: str = 'train',
                 train_ratio: float = 0.9,
                 max_samples_per_graph: int = 2000,
                 seed: Optional[int] = None):
        super().__init__()
        self.csv_paths = csv_paths
        self.split = split
        self.train_ratio = train_ratio
        self.max_samples_per_graph = max_samples_per_graph

        if seed is not None:
            np.random.seed(seed)

        self.graph_metadata = self._process_csvs()

    def _process_csvs(self) -> List[Dict]:
        metadata = []

        for csv_path in self.csv_paths:
            df = pd.read_csv(csv_path)
            max_serial = df['serial'].max()
            valid_nodes = df[df['serial'] < MAX_SERIAL]['serial'].values

            if len(valid_nodes) == 0:
                continue

            # Set number of samples as minimum between node count and max samples
            total_samples = min(len(valid_nodes), self.max_samples_per_graph)

            # Split samples between train and test
            if self.split == 'train':
                n_samples = int(total_samples * self.train_ratio)
            else:
                n_samples = total_samples - int(total_samples * self.train_ratio)

            metadata.append({
                'csv_path': csv_path,
                'max_serial': max_serial,
                'valid_nodes': valid_nodes,
                'n_samples': n_samples
            })

        return metadata

    def process_node(self, csv_path: str, center_node_serial: int, max_serial: int) -> Optional[Data]:
        df = pd.read_csv(csv_path)

        valid_nodes = df[
            (df['serial'] <= center_node_serial) &
            (df['serial'] <= MAX_SERIAL)
        ]

        if valid_nodes.empty:
            return None

        node_mapping = {
            serial: idx for idx, serial
            in enumerate(valid_nodes['serial'].values)
        }

        node_features = valid_nodes[[
            'f', 'h', 'g', 'BF', 'h0', 'H_min',
            'last_H_min_update', 'f_max'
        ]].values

        edge_list = []
        for _, row in valid_nodes.iterrows():
            if pd.notna(row['father_serial']):
                father_serial = int(row['father_serial'])
                if father_serial in node_mapping:
                    edge_list.extend([
                        [node_mapping[father_serial], node_mapping[row['serial']]],
                        [node_mapping[row['serial']], node_mapping[father_serial]]
                    ])

        if not edge_list:
            return None

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.tensor(valid_nodes['serial'].values / max_serial, dtype=torch.float)
        center_idx = node_mapping[center_node_serial]
        original_serials = torch.tensor(valid_nodes['serial'].values, dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            center_idx=center_idx,
            original_serials=original_serials
        )

    def len(self):
        return sum(meta['n_samples'] for meta in self.graph_metadata)

    def get(self, idx):
        current_count = 0
        for meta in self.graph_metadata:
            if idx < current_count + meta['n_samples']:
                while True:
                    center_node_serial = np.random.choice(meta['valid_nodes'])
                    graph = self.process_node(
                        meta['csv_path'],
                        center_node_serial,
                        meta['max_serial']
                    )
                    if graph is not None:
                        return graph

                print(f"Warning: Could not create valid graph for {meta['csv_path']}")
                return self.get((idx + 1) % self.len())

            current_count += meta['n_samples']
        raise IndexError("Index out of range")

def create_and_save_dataloaders(
    csv_paths: List[str],
    output_dir: Path,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    max_samples_per_graph: int = 2000,
    seed: Optional[int] = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'full_clipped_graphs_without_future_multi_target.pt'

    metadata = {
        'csv_paths': csv_paths,
        'batch_size': batch_size,
        'train_ratio': train_ratio,
        'max_samples_per_graph': max_samples_per_graph,
        'seed': seed
    }
    torch.save(metadata, save_path)

    train_dataset = EfficientSearchGraphDataset(
        csv_paths=csv_paths,
        split='train',
        train_ratio=train_ratio,
        max_samples_per_graph=max_samples_per_graph,
        seed=seed
    )

    test_dataset = EfficientSearchGraphDataset(
        csv_paths=csv_paths,
        split='test',
        train_ratio=train_ratio,
        max_samples_per_graph=max_samples_per_graph,
        seed=seed
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_dataloaders(dataset_path: Path, batch_size: Optional[int] = None):
    metadata = torch.load(dataset_path)
    batch_size = batch_size or metadata['batch_size']

    train_dataset = EfficientSearchGraphDataset(
        csv_paths=metadata['csv_paths'],
        split='train',
        train_ratio=metadata['train_ratio'],
        max_samples_per_graph=metadata['max_samples_per_graph'],
        seed=metadata['seed']
    )

    test_dataset = EfficientSearchGraphDataset(
        csv_paths=metadata['csv_paths'],
        split='test',
        train_ratio=metadata['train_ratio'],
        max_samples_per_graph=metadata['max_samples_per_graph'],
        seed=metadata['seed']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    root_dir = Path(__file__).resolve().parent.parent
    csv_dir = root_dir / 'csv_output'
    dataset_dir = root_dir / 'processed_datasets'

    csv_paths = list(csv_dir.glob('*.csv'))
    print(f"Found {len(csv_paths)} CSV files")

    params = {
        'batch_size': 32,
        'train_ratio': 0.8,
        'max_samples_per_graph': 20,
        'seed': 42
    }

    print("Creating and saving dataloaders...")
    create_and_save_dataloaders(
        csv_paths=csv_paths,
        output_dir=dataset_dir,
        **params
    )
    print(f"Saved dataloaders to {dataset_dir}")

if __name__ == '__main__':
    main()