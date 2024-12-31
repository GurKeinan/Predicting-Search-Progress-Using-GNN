import torch
from torch_geometric.data import Data, Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List, Optional, Dict
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

MAX_SERIAL = 30000

def process_single_node(args):
    csv_path, center_node_serial, max_serial = args

    try:
        df = pd.read_csv(csv_path)

        # Filter nodes
        valid_nodes = df[
            (df['serial'] <= center_node_serial) &
            (df['serial'] <= MAX_SERIAL)
        ]

        if valid_nodes.empty:
            return None

        # Create node mapping for the filtered subgraph
        node_mapping = {
            serial: idx for idx, serial
            in enumerate(valid_nodes['serial'].values)
        }

        # Extract node features
        node_features = valid_nodes[[
            'f', 'h', 'g', 'BF', 'h0', 'H_min',
            'last_H_min_update', 'f_max'
        ]].values

        # Create edges (bidirectional)
        edge_list = []
        for _, row in valid_nodes.iterrows():
            if pd.notna(row['father_serial']):
                father_serial = int(row['father_serial'])
                if father_serial in node_mapping:  # Check if father is in subgraph
                    # Add both directions
                    edge_list.extend([
                        [node_mapping[father_serial], node_mapping[row['serial']]],
                        [node_mapping[row['serial']], node_mapping[father_serial]]
                    ])

        if not edge_list:  # If no edges, skip this graph
            return None

        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create target for all nodes
        y = torch.tensor(
            valid_nodes['serial'].values / max_serial,
            dtype=torch.float
        )

        # Store the index of the center node in our subgraph
        center_idx = node_mapping[center_node_serial]

        # Store the original serial numbers for later use
        original_serials = torch.tensor(
            valid_nodes['serial'].values,
            dtype=torch.long
        )

        return {
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'center_idx': center_idx,
            'original_serials': original_serials
        }

    except Exception as e:
        print(f"Error processing node {center_node_serial}: {str(e)}")
        return None

class NewSearchGraphDataset(Dataset):
    def __init__(self,
                 csv_paths: List[str],
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 seed: Optional[int] = None,
                 num_workers: int = None):
        super().__init__()
        self.csv_paths = csv_paths
        self.split = split
        self.train_ratio = train_ratio
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.8))

        if seed is not None:
            np.random.seed(seed)

        self.data_list = self._process_files()

    def _process_csv_file(self, csv_path: str) -> List[Data]:
        graphs = []
        df = pd.read_csv(csv_path)

        # Get max serial for the whole graph
        max_serial = df['serial'].max()

        # First, filter out nodes with serial >= MAX_SERIAL
        valid_nodes = df[df['serial'] < MAX_SERIAL]['serial'].values
        n_nodes = len(valid_nodes)

        if n_nodes == 0:
            return []

        # Determine desired sample sizes
        total_desired_samples = 2000  # Total samples we want
        if self.split == 'train':
            desired_samples = int(total_desired_samples * self.train_ratio)
        else:
            desired_samples = total_desired_samples - int(total_desired_samples * self.train_ratio)

        # Sample with replacement if we need more samples than available nodes
        sampled_serials = np.random.choice(
            valid_nodes,
            size=desired_samples,
            replace=(desired_samples > n_nodes)
        )

        selected_serials = sampled_serials  # No need to split since we already sampled the correct amount

        process_args = [
            (csv_path, int(serial), max_serial)
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
                            y=result['y'],
                            center_idx=result['center_idx'],
                            original_serials=result['original_serials']
                        )
                        graphs.append(graph)
                    pbar.update(1)

        return graphs

    def _process_files(self):
        all_data = []
        with tqdm(total=len(self.csv_paths),
                 desc=f'Processing {self.split} set CSV files') as pbar:
            for csv_path in self.csv_paths:
                graphs = self._process_csv_file(csv_path)
                all_data.extend(graphs)
                pbar.update(1)
        return all_data

    def save(self, path: str):
        """Save the dataset to disk"""
        save_dict = {
            'data_list': self.data_list,
            'csv_paths': self.csv_paths,
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
            split=save_dict['split'],
            train_ratio=save_dict['train_ratio']
        )
        dataset.data_list = save_dict['data_list']
        return dataset

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def create_and_save_datasets(
    csv_paths: List[str],
    output_dir: Path,
    train_ratio: float = 0.8,
    num_workers: int = None,
    seed: Optional[int] = None
):
    """Create and save the datasets"""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = NewSearchGraphDataset(
        csv_paths, 'train', train_ratio, seed, num_workers
    )
    test_dataset = NewSearchGraphDataset(
        csv_paths, 'test', train_ratio, seed, num_workers
    )

    train_dataset.save(output_dir / 'train_dataset_full_graph_without_future.pkl')
    test_dataset.save(output_dir / 'test_dataset_full_graph_without_future.pkl')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Saved datasets to {output_dir}")

def load_and_create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32
):
    """Load datasets and create DataLoader objects"""
    train_dataset = NewSearchGraphDataset.load(dataset_dir / 'train_dataset_full_graph_without_future.pkl')
    test_dataset = NewSearchGraphDataset.load(dataset_dir / 'test_dataset_full_graph_without_future.pkl')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

from pathlib import Path

def main():
    # Set up paths
    root_dir = Path(__file__).resolve().parent.parent  # Go up one directory
    csv_dir = root_dir / 'csv_output_mini'  # Directory containing CSV files
    output_dir = root_dir / 'processed_datasets'  # Directory to save processed datasets

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all CSV files from the directory
    csv_paths = list(csv_dir.glob('*.csv'))
    print(f"Found {len(csv_paths)} CSV files in {csv_dir}")

    # Parameters
    train_ratio = 0.8
    seed = 42
    num_workers = None  # Will use 80% of CPU cores by default

    # Create and save datasets
    create_and_save_datasets(
        csv_paths=csv_paths,
        output_dir=output_dir,
        train_ratio=train_ratio,
        num_workers=num_workers,
        seed=seed
    )

    print("Dataset creation completed!")

if __name__ == '__main__':
    main()