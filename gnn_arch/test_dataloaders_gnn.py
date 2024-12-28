import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import GraphNorm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader
import sys

# Add the directory containing k_hop_without_future.py to the Python path
root_dir = Path(__file__).resolve().parent.parent
creating_trees_dir = root_dir / 'creating_trees_from_csv'
sys.path.append(str(creating_trees_dir))

# Now we can import from k_hop_without_future
from k_hop_without_future import load_and_create_dataloaders

class SearchGraphGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=4):
        super(SearchGraphGNN, self).__init__()
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

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, batch, center_idx):
        # Initial convolution
        x = self.conv_first(x, edge_index)
        x = self.norm_first(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Middle convolutions
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new, batch)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x + x_new  # Skip connection

        # Final convolution
        x = self.conv_last(x, edge_index)
        x = self.norm_last(x, batch)
        x = F.relu(x)

        # Global pooling
        x_global = global_mean_pool(x, batch)

        # Get center node features
        center_features = x[center_idx]

        # Concatenate global and center features
        x_combined = torch.cat([x_global, center_features], dim=1)

        # Final prediction
        out = self.mlp(x_combined)
        return out.squeeze()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc='Training', leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index, data.batch, data.center_idx)
        loss = criterion(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    for data in tqdm(loader, desc='Evaluating', leave=False):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.center_idx)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs

        predictions.extend(out.cpu().numpy())
        targets.extend(data.y.cpu().numpy())

    mse = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    r2 = np.corrcoef(predictions, targets)[0, 1] ** 2

    return mse, mae, r2, predictions, targets

def plot_training_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_prediction_scatter(predictions, targets, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_model_and_results(model, output_dir, train_losses, val_losses,
                          predictions, targets, metrics):
    """Save model, plots, and metrics"""
    # Save model
    torch.save(model.state_dict(), output_dir / 'best_model.pt')

    # Save plots
    plot_training_curves(train_losses, val_losses, output_dir / 'training_curves.png')
    plot_prediction_scatter(predictions, targets, output_dir / 'predictions.png')

    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value}\n')

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent
    dataset_dir = root_dir / 'dataloaders_datasets'
    output_dir = root_dir / 'model_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data using our new loader
    print("Loading datasets and creating dataloaders...")
    train_loader, test_loader = load_and_create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=32
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Model parameters
    num_node_features = train_loader.dataset[0].x.shape[1]
    hidden_channels = 64
    num_layers = 4

    # Create model
    print("Creating model...")
    model = SearchGraphGNN(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)

    # Training parameters
    learning_rate = 0.001
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 10
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mae, val_r2, _, _ = evaluate(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print metrics
        print(f'Epoch {epoch+1:03d}, Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model state
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model and evaluate
    print("Loading best model and evaluating...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_loss, test_mae, test_r2, predictions, targets = evaluate(
        model, test_loader, criterion, device
    )

    # Save final results
    metrics = {
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    save_model_and_results(
        model, output_dir, train_losses, val_losses, predictions, targets, metrics
    )

    print("Final Results:")
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R2: {test_r2:.4f}')
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()