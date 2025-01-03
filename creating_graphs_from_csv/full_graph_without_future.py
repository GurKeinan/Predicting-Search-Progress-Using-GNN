import torch
from torch.utils.data.sampler import BatchSampler
from torch_geometric.data import Data, Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Iterator, Tuple
from collections import defaultdict
import pickle

MAX_SERIAL = 20000

class LazySearchGraphDataset(Dataset):
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

        self.file_metadata = self._get_file_metadata()
        self.length = self._calculate_total_samples()

        self._df_cache = {}
        self._cache_size = 3

    def _get_file_metadata(self) -> List[Dict]:
        metadata = []
        for csv_path in self.csv_paths:
            df_info = pd.read_csv(csv_path, usecols=['serial'])
            valid_node_count = len(df_info[df_info['serial'] < MAX_SERIAL])

            if valid_node_count > 0:
                samples = min(valid_node_count, self.max_samples_per_graph)
                if self.split == 'train':
                    n_samples = int(samples * self.train_ratio)
                else:
                    n_samples = samples - int(samples * self.train_ratio)

                metadata.append({
                    'path': str(csv_path),  # Convert Path to str for serialization
                    'n_samples': n_samples,
                    'valid_node_count': valid_node_count
                })
        return metadata

    def _calculate_total_samples(self) -> int:
        return sum(meta['n_samples'] for meta in self.file_metadata)

    def _get_dataframe(self, csv_path: str) -> pd.DataFrame:
        if csv_path not in self._df_cache:
            if len(self._df_cache) >= self._cache_size:
                lru_key = next(iter(self._df_cache))
                del self._df_cache[lru_key]

            self._df_cache[csv_path] = pd.read_csv(csv_path)

        df = self._df_cache.pop(csv_path)
        self._df_cache[csv_path] = df
        return df

    def _get_file_for_index(self, idx: int) -> Dict:
        current_count = 0
        for meta in self.file_metadata:
            next_count = current_count + meta['n_samples']
            if idx < next_count:
                return meta
            current_count = next_count
        raise IndexError("Index out of range")

    def process_node(self, df: pd.DataFrame, center_node_serial: int) -> Optional[Data]:
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
        y = torch.tensor(valid_nodes['serial'].values / MAX_SERIAL, dtype=torch.float)
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
        return self.length

    def get(self, idx):
        file_meta = self._get_file_for_index(idx)
        df = self._get_dataframe(file_meta['path'])

        valid_nodes = df[df['serial'] < MAX_SERIAL]['serial'].values
        while True:
            center_node_serial = np.random.choice(valid_nodes)
            graph = self.process_node(df, center_node_serial)
            if graph is not None:
                return graph

def create_and_save_dataloaders(
    csv_paths: List[str],
    output_dir: Path,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    max_samples_per_graph: int = 2000,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'dataset_metadata.pt'

    metadata = {
        'csv_paths': [str(path) for path in csv_paths],
        'batch_size': batch_size,
        'train_ratio': train_ratio,
        'max_samples_per_graph': max_samples_per_graph,
        'seed': seed
    }
    torch.save(metadata, save_path)

    train_dataset = LazySearchGraphDataset(
        csv_paths=csv_paths,
        split='train',
        train_ratio=train_ratio,
        max_samples_per_graph=max_samples_per_graph,
        seed=seed
    )

    test_dataset = LazySearchGraphDataset(
        csv_paths=csv_paths,
        split='test',
        train_ratio=train_ratio,
        max_samples_per_graph=max_samples_per_graph,
        seed=seed
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_dataloaders(dataset_path: Path, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    metadata = torch.load(dataset_path)
    batch_size = batch_size or metadata['batch_size']

    train_dataset = LazySearchGraphDataset(
        csv_paths=[Path(p) for p in metadata['csv_paths']],
        split='train',
        train_ratio=metadata['train_ratio'],
        max_samples_per_graph=metadata['max_samples_per_graph'],
        seed=metadata['seed']
    )

    test_dataset = LazySearchGraphDataset(
        csv_paths=[Path(p) for p in metadata['csv_paths']],
        split='test',
        train_ratio=metadata['train_ratio'],
        max_samples_per_graph=metadata['max_samples_per_graph'],
        seed=metadata['seed']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent.parent
    csv_dir = root_dir / 'csv_output'
    dataset_dir = root_dir / 'processed_datasets'

    csv_paths = list(csv_dir.glob('*.csv'))
    print(f"Found {len(csv_paths)} CSV files")

    train_loader, test_loader = create_and_save_dataloaders(
        csv_paths=csv_paths,
        output_dir=dataset_dir,
        batch_size=32,
        train_ratio=0.8,
        max_samples_per_graph=20,
        seed=42
    )

    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Testing dataset size: {len(test_loader.dataset)}")