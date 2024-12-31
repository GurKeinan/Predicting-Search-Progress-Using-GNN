import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import sys

# Add the parent directory to the Python path
parent_path = Path(__file__).resolve().parent.parent
creating_graphs_from_csv_dir_path = parent_path / 'creating_graphs_from_csv'
sys.path.append(str(creating_graphs_from_csv_dir_path))

from full_graph_without_future import load_and_create_dataloaders

def convert_to_serializable(obj):
    """Convert a value to a JSON-serializable format."""
    # Handle None
    if obj is None:
        return None

    # Handle basic numeric types
    if hasattr(obj, 'dtype'):
        # Convert any NumPy number to a Python number
        return obj.item() if obj.ndim == 0 else obj.tolist()

    # Handle torch tensors
    if torch.is_tensor(obj):
        return obj.cpu().detach().numpy().tolist()

    # Handle dictionaries
    if isinstance(obj, (dict, map)):
        return {key: convert_to_serializable(value) for key, value in obj.items()}

    # Handle lists, tuples, and sets
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(item) for item in obj]

    # Handle anything else - try direct conversion or return as is
    try:
        return float(obj) if isinstance(obj, (int, float)) else obj
    except (TypeError, ValueError):
        return obj
class MultiTargetSearchGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=4):
        super(MultiTargetSearchGNN, self).__init__()
        self.num_layers = num_layers

        # Initial convolution
        self.conv_first = GCNConv(num_node_features, hidden_channels)
        self.norm_first = GraphNorm(hidden_channels)

        # Middle convolution layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 2)
        ])
        self.norms = nn.ModuleList([
            GraphNorm(hidden_channels)
            for _ in range(num_layers - 2)
        ])

        # Final convolution
        self.conv_last = GCNConv(hidden_channels, hidden_channels)
        self.norm_last = GraphNorm(hidden_channels)

        # MLP for node-level predictions with sigmoid activation
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Added sigmoid activation
        )

    def _get_node_embeddings(self, x, edge_index, batch):
        # Initial convolution
        x = self.conv_first(x, edge_index)
        x = self.norm_first(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Middle convolutions with skip connections
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new, batch)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x + x_new

        # Final convolution
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

            # Since predictions are now between 0 and 1, we need to handle division differently
            # to avoid division by very small numbers
            epsilon = 1e-6  # Small constant to prevent division by zero
            implied_max_serials = graph_serials / (graph_predictions + epsilon)
            mean_implied_max_serial = torch.mean(implied_max_serials)

            center_idx = data.center_idx[i]
            center_serial = graph_serials[center_idx].to(device)

            final_prediction = center_serial / mean_implied_max_serial

            # Clip the final prediction to ensure it's between 0 and 1
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

    mse = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(all_predictions - all_targets))
    r2 = np.corrcoef(all_predictions, all_targets)[0, 1] ** 2

    center_mse = np.mean((all_center_predictions - all_center_targets) ** 2)
    center_mae = np.mean(np.abs(all_center_predictions - all_center_targets))
    center_r2 = np.corrcoef(all_center_predictions, all_center_targets)[0, 1] ** 2

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'center_mse': center_mse,
        'center_mae': center_mae,
        'center_r2': center_r2
    }

    return metrics, all_center_predictions, all_center_targets

def train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001,
                device='cuda', patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_metrics_history = []
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        val_metrics, _, _ = evaluate(model, test_loader, criterion, device)
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
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_metrics_history

def save_experiment_info(params, metrics, output_dir: Path):
    info = {
        'parameters': convert_to_serializable(params),
        'final_metrics': convert_to_serializable(metrics),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'experiment_info.json', 'w') as f:
        json.dump(info, f, indent=4)

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    root_dir = Path(__file__).resolve().parent.parent
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
        'seed': 42
    }

    set_random_seeds(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_loader, test_loader = load_and_create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=params['batch_size']
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    print("Creating model...")
    num_node_features = train_loader.dataset[0].x.shape[1]
    model = MultiTargetSearchGNN(
        num_node_features=num_node_features,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers']
    ).to(device)

    print("Starting training...")
    model, train_losses, val_metrics_history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=params['epochs'],
        lr=params['learning_rate'],
        device=device,
        patience=params['patience']
    )

    print("Performing final evaluation...")
    final_metrics, predictions, targets = evaluate(
        model, test_loader, nn.MSELoss(), device
    )

    print("Saving results...")
    torch.save(model.state_dict(), output_dir / 'model.pt')
    save_experiment_info(params, final_metrics, output_dir)

    print("\nFinal Results:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()