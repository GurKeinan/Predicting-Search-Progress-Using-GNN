import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import GCNConv, GraphNorm
import sys

root_dir = Path(__file__).resolve().parent.parent
creating_graphs_from_csv = root_dir / 'creating_graphs_from_csv'
sys.path.append(str(creating_graphs_from_csv))
from full_graph_without_future import create_and_save_dataloaders, load_dataloaders

class MultiTargetSearchGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=4):
        super(MultiTargetSearchGNN, self).__init__()
        self.num_layers = num_layers

        self.conv_first = GCNConv(num_node_features, hidden_channels)
        self.norm_first = GraphNorm(hidden_channels)

        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 2)
        ])
        self.norms = nn.ModuleList([
            GraphNorm(hidden_channels)
            for _ in range(num_layers - 2)
        ])

        self.conv_last = GCNConv(hidden_channels, hidden_channels)
        self.norm_last = GraphNorm(hidden_channels)

        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def _get_node_embeddings(self, x, edge_index, batch):
        x = self.conv_first(x, edge_index)
        x = self.norm_first(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new, batch)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x + x_new

        x = self.conv_last(x, edge_index)
        x = self.norm_last(x, batch)
        x = F.relu(x)
        return x

    def forward(self, x, edge_index, batch):
        x = self._get_node_embeddings(x, edge_index, batch)
        predictions = self.node_predictor(x).squeeze(-1)
        return predictions

    def predict(self, data):
        device = next(self.parameters()).device
        data = data.to(device)
        all_node_predictions = self(data.x, data.edge_index, data.batch)

        final_predictions = []
        batch_size = len(data.ptr) - 1

        for i in range(batch_size):
            start_idx = data.ptr[i].item()
            end_idx = data.ptr[i + 1].item()

            graph_predictions = all_node_predictions[start_idx:end_idx]
            graph_serials = data.original_serials[start_idx:end_idx].to(device).float()

            epsilon = 1e-6
            implied_max_serials = graph_serials / (graph_predictions + epsilon)
            mean_implied_max_serial = torch.mean(implied_max_serials)

            center_idx = data.center_idx[i]
            center_serial = graph_serials[center_idx].to(device)

            final_prediction = center_serial / mean_implied_max_serial
            final_prediction = torch.clamp(final_prediction, 0.0, 1.0)
            final_predictions.append(final_prediction)

        return torch.stack(final_predictions)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc='Training', leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_center_predictions = []
    all_center_targets = []

    for data in tqdm(loader, desc='Evaluating', leave=False):
        data = data.to(device)
        all_node_predictions = model(data.x, data.edge_index, data.batch)
        loss = criterion(all_node_predictions, data.y)
        total_loss += loss.item() * data.num_graphs

        center_predictions = model.predict(data)

        batch_size = len(data.ptr) - 1
        for i in range(batch_size):
            start_idx = data.ptr[i].item()
            center_idx = data.center_idx[i] + start_idx
            end_idx = data.ptr[i + 1].item()

            all_center_predictions.append(center_predictions[i].cpu())
            all_center_targets.append(data.y[center_idx].cpu())
            all_predictions.extend(all_node_predictions[start_idx:end_idx].cpu().numpy())
            all_targets.extend(data.y[start_idx:end_idx].cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_center_predictions = torch.stack(all_center_predictions).numpy()
    all_center_targets = torch.stack(all_center_targets).numpy()

    metrics = {
        'mse': total_loss / len(loader.dataset),
        'mae': np.mean(np.abs(all_predictions - all_targets)),
        'r2': np.corrcoef(all_predictions, all_targets)[0, 1] ** 2,
        'center_mse': np.mean((all_center_predictions - all_center_targets) ** 2),
        'center_mae': np.mean(np.abs(all_center_predictions - all_center_targets)),
        'center_r2': np.corrcoef(all_center_predictions, all_center_targets)[0, 1] ** 2
    }

    return metrics, all_center_predictions, all_center_targets

def save_experiment_info(params, metrics, output_dir: Path):
    info = {
        'parameters': params,
        'final_metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'experiment_info.json', 'w') as f:
        json.dump(info, f, indent=4)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    csv_dir = root_dir / 'csv_output'
    dataset_dir = root_dir / 'processed_datasets'
    output_dir = root_dir / 'model_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        'hidden_channels': 64,
        'num_layers': 4,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
        'seed': 42,
        'train_ratio': 0.8,
        'max_samples_per_graph': 2000
    }

    if params['seed'] is not None:
        torch.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        np.random.seed(params['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader_path = dataset_dir / 'dataset_metadata.pt'
    if dataloader_path.exists():
        print("Loading existing dataloaders...")
        train_loader, test_loader = load_dataloaders(dataloader_path, params['batch_size'])
    else:
        print("Creating new dataloaders...")
        csv_paths = list(csv_dir.glob('*.csv'))
        train_loader, test_loader = create_and_save_dataloaders(
            csv_paths=csv_paths,
            output_dir=dataset_dir,
            batch_size=params['batch_size'],
            train_ratio=params['train_ratio'],
            max_samples_per_graph=params['max_samples_per_graph'],
            seed=params['seed']
        )

    print("Creating model...")
    num_node_features = train_loader.dataset[0].x.shape[1]
    model = MultiTargetSearchGNN(
        num_node_features=num_node_features,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_metrics_history = []
    best_model_state = None

    print("Starting training...")
    for epoch in range(params['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)

        print(f'Epoch {epoch+1:03d}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Center MSE: {val_metrics["center_mse"]:.4f}')
        print(f'Val Center MAE: {val_metrics["center_mae"]:.4f}')
        print(f'Val Center R2: {val_metrics["center_r2"]:.4f}')

        scheduler.step(val_metrics['center_mse'])

        if val_metrics['center_mse'] < best_val_loss:
            best_val_loss = val_metrics['center_mse']
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    final_metrics, predictions, targets = evaluate(model, test_loader, criterion, device)

    print("Saving results...")
    torch.save(model.state_dict(), output_dir / 'model.pt')
    save_experiment_info(params, final_metrics, output_dir)

    print("\nFinal Results:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()