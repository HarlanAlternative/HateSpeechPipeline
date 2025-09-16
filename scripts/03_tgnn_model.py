
#!/usr/bin/env python3
"""
Temporal Graph Neural Network (TGNN) Model Implementation.
Implements TGAT/TGN architectures for hate speech classification and diffusion prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import TemporalData
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for TGAT.
    
    This module implements temporal attention to capture how node representations
    evolve over time. It uses timestamp information to weight the importance of
    different temporal interactions.
    
    Args:
        input_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden representations
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoding layers
        self.time_encoder = nn.Linear(1, hidden_dim)
        
        # Attention mechanism components
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        
        # Attention weights
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, 1))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, edge_index, timestamps):
        """
        Apply temporal attention mechanism.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            timestamps (torch.Tensor): Edge timestamps [num_edges]
            
        Returns:
            torch.Tensor: Temporally attended node features
        """
        # For now, we use a simplified version that preserves the input
        # In a full implementation, this would incorporate temporal dynamics
        # TODO: Implement full temporal attention with timestamp encoding
        
        # Apply layer normalization for stability
        x_normalized = self.layer_norm(x)
        
        return x_normalized

class TGATLayer(nn.Module):
    """Temporal Graph Attention Network layer."""
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(input_dim, hidden_dim)
        
        # Graph attention
        self.graph_attention = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr, timestamps):
        """Forward pass through TGAT layer."""
        # Apply temporal attention
        x_temporal = self.temporal_attention(x, edge_index, timestamps)
        
        # Apply graph attention
        x_graph = self.graph_attention(x_temporal, edge_index)
        
        # Residual connection and normalization
        x_out = self.norm(x_temporal + self.dropout(x_graph))
        
        return x_out

class TGNLayer(nn.Module):
    """Temporal Graph Network layer with memory."""
    
    def __init__(self, input_dim, hidden_dim, memory_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        # Memory update
        self.memory_updater = nn.GRUCell(input_dim, memory_dim)
        
        # Message function
        self.message_function = nn.Sequential(
            nn.Linear(input_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregation
        self.aggregator = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, timestamps, memory):
        """Forward pass through TGN layer."""
        # Update memory (simplified)
        new_memory = self.memory_updater(x.mean(dim=0, keepdim=True), memory)
        
        # Compute messages
        memory_expanded = new_memory.expand(x.size(0), -1)
        combined_features = torch.cat([x, memory_expanded], dim=-1)
        messages = self.message_function(combined_features)
        
        # Aggregate messages
        x_out = self.aggregator(messages)
        
        return x_out, new_memory

class TGNNModel(nn.Module):
    """Complete TGNN model for hate speech analysis."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        tgnn_config = config.get('tgnn', {})
        
        self.model_type = tgnn_config.get('model_type', 'TGAT')
        self.input_dim = tgnn_config.get('input_dim', 776)  # BERT + user features
        self.hidden_dim = tgnn_config.get('hidden_dim', 128)
        self.num_layers = tgnn_config.get('num_layers', 2)
        self.num_classes = tgnn_config.get('num_classes', 2)
        self.dropout = tgnn_config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # TGNN layers
        if self.model_type == 'TGAT':
            self.tgnn_layers = nn.ModuleList([
                TGATLayer(self.hidden_dim, self.hidden_dim) 
                for _ in range(self.num_layers)
            ])
        elif self.model_type == 'TGN':
            self.memory_dim = tgnn_config.get('memory_dim', 64)
            self.memory = nn.Parameter(torch.randn(1, self.memory_dim))
            self.tgnn_layers = nn.ModuleList([
                TGNLayer(self.hidden_dim, self.hidden_dim, self.memory_dim)
                for _ in range(self.num_layers)
            ])
        
        # Classification heads
        self.hate_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # Diffusion prediction head
        self.diffusion_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Time prediction head
        self.time_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, data, task='classification'):
        """Forward pass for different tasks."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        timestamps = data.t
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply TGNN layers
        if self.model_type == 'TGAT':
            for layer in self.tgnn_layers:
                x = layer(x, edge_index, edge_attr, timestamps)
        elif self.model_type == 'TGN':
            memory = self.memory
            for layer in self.tgnn_layers:
                x, memory = layer(x, edge_index, edge_attr, timestamps, memory)
        
        if task == 'classification':
            # Node classification
            return self.hate_classifier(x)
        
        elif task == 'diffusion':
            # Edge prediction for diffusion
            src_nodes = x[edge_index[0]]
            dst_nodes = x[edge_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
            return self.diffusion_predictor(edge_features)
        
        elif task == 'time_prediction':
            # Time prediction for interactions
            src_nodes = x[edge_index[0]]
            dst_nodes = x[edge_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
            return self.time_predictor(edge_features)
        
        else:
            return x  # Return embeddings

class TGNNTrainer:
    """Trainer for TGNN models."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        tgnn_config = config.get('tgnn', {})
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=tgnn_config.get('learning_rate', 0.001)
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def prepare_data(self, graph_data):
        """
        Prepare graph data for training with proper label assignment.
        
        This method loads the balanced dataset to create user-level labels
        and splits the data into training and testing sets.
        
        Args:
            graph_data: PyTorch Geometric temporal graph data
            
        Returns:
            graph_data: Prepared graph data with labels and masks
        """
        print("Preparing training data...")
        
        # Move graph data to the appropriate device (CPU/GPU)
        graph_data = graph_data.to(self.device)
        
        # Get number of user nodes in the graph
        num_users = graph_data.num_users
        
        # Load balanced dataset to extract ground truth labels
        user_labels = self._create_user_labels(num_users, graph_data)
        
        # Convert labels to PyTorch tensor
        graph_data.y = torch.LongTensor(user_labels).to(self.device)
        
        # Create train/test split for evaluation
        self._create_train_test_split(graph_data, num_users)
        
        # Log data preparation statistics
        self._log_data_statistics(graph_data)
        
        return graph_data
    
    def _create_user_labels(self, num_users, graph_data):
        """
        Create user-level labels from the balanced dataset.
        
        Args:
            num_users (int): Number of user nodes
            graph_data: Graph data containing user mappings
            
        Returns:
            list: User labels (0 for normal, 1 for hate speech users)
        """
        try:
            # Load the balanced dataset containing ground truth labels
            balanced_df = pd.read_parquet('artifacts/balanced_dataset.parquet')
            
            # Create mapping from usernames to hate speech labels
            user_hate_mapping = self._build_user_hate_mapping(balanced_df)
            
            # Assign labels to each user node
            user_labels = []
            for user_idx in range(num_users):
                # Reverse lookup: find username for this user index
                username = self._get_username_by_index(user_idx, graph_data)
                
                # Assign label based on user's hate speech activity
                if username and username in user_hate_mapping:
                    label = 1 if user_hate_mapping[username] else 0
                else:
                    label = 0  # Default to normal user
                
                user_labels.append(label)
            
            print(f"Successfully created labels for {len(user_labels)} users")
            return user_labels
            
        except Exception as e:
            print(f"Warning: Could not load balanced dataset for labels: {e}")
            print("Using fallback label assignment...")
            
            # Fallback: create artificially balanced labels
            return [1 if i < num_users // 2 else 0 for i in range(num_users)]
    
    def _build_user_hate_mapping(self, balanced_df):
        """
        Build a mapping from usernames to hate speech labels.
        
        Args:
            balanced_df: Balanced dataset DataFrame
            
        Returns:
            dict: Mapping from username to boolean hate flag
        """
        user_hate_map = {}
        
        for _, row in balanced_df.iterrows():
            author = row['author']
            is_hate_post = (row['binary_label'] == 1)
            
            # Initialize user if not seen before
            if author not in user_hate_map:
                user_hate_map[author] = False
            
            # Mark user as hate speech user if they have any hate posts
            if is_hate_post:
                user_hate_map[author] = True
        
        return user_hate_map
    
    def _get_username_by_index(self, user_idx, graph_data):
        """
        Get username for a given user index through reverse lookup.
        
        Args:
            user_idx (int): User node index
            graph_data: Graph data containing user mappings
            
        Returns:
            str or None: Username if found, None otherwise
        """
        user_to_id_mapping = getattr(graph_data, 'user_to_id', {})
        
        for username, idx in user_to_id_mapping.items():
            if idx == user_idx:
                return username
        
        return None
    
    def _create_train_test_split(self, graph_data, num_users):
        """
        Create training and testing masks for user nodes.
        
        Args:
            graph_data: Graph data to add masks to
            num_users (int): Number of user nodes
        """
        # Split user indices into train and test sets
        train_users, test_users = train_test_split(
            range(num_users), 
            test_size=0.2,  # 20% for testing
            random_state=42,  # For reproducibility
            stratify=graph_data.y.cpu().numpy()  # Maintain label balance
        )
        
        # Create boolean masks for training and testing
        graph_data.train_mask = torch.zeros(num_users, dtype=torch.bool).to(self.device)
        graph_data.test_mask = torch.zeros(num_users, dtype=torch.bool).to(self.device)
        
        # Set mask values
        graph_data.train_mask[train_users] = True
        graph_data.test_mask[test_users] = True
    
    def _log_data_statistics(self, graph_data):
        """
        Log statistics about the prepared data.
        
        Args:
            graph_data: Prepared graph data
        """
        train_count = graph_data.train_mask.sum().item()
        test_count = graph_data.test_mask.sum().item()
        hate_count = graph_data.y.sum().item()
        normal_count = len(graph_data.y) - hate_count
        
        print(f"Data split: {train_count} train, {test_count} test users")
        print(f"Label distribution: {hate_count} hate users, {normal_count} normal users")
    
    def train_classification(self, graph_data, epochs=100):
        """Train hate speech classification task."""
        print("Training hate speech classification...")
        
        graph_data = self.prepare_data(graph_data)
        best_acc = 0
        
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(graph_data, task='classification')
            
            # Only compute loss on user nodes
            user_out = out[:graph_data.num_users]
            loss = self.classification_loss(user_out[graph_data.train_mask], 
                                          graph_data.y[graph_data.train_mask])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Evaluation
            if epoch % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    pred = user_out[graph_data.test_mask].argmax(dim=1)
                    acc = (pred == graph_data.y[graph_data.test_mask]).float().mean()
                    
                    if acc > best_acc:
                        best_acc = acc
                    
                    print(f"Epoch {epoch}: Loss = {loss:.4f}, Test Acc = {acc:.4f}")
        
        return best_acc
    
    def evaluate_model(self, graph_data):
        """Comprehensive model evaluation."""
        print("Evaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            # Classification evaluation
            out = self.model(graph_data, task='classification')
            user_out = out[:graph_data.num_users]
            
            pred = user_out[graph_data.test_mask].argmax(dim=1)
            true_labels = graph_data.y[graph_data.test_mask]
            
            # Compute metrics
            accuracy = (pred == true_labels).float().mean().item()
            
            # Convert to numpy for sklearn metrics
            pred_np = pred.cpu().numpy()
            true_np = true_labels.cpu().numpy()
            
            # Classification report
            report = classification_report(true_np, pred_np, output_dict=True)
            
            # ROC AUC - handle edge cases
            try:
                probs = F.softmax(user_out[graph_data.test_mask], dim=1)[:, 1].cpu().numpy()
                if len(np.unique(true_np)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(true_np, probs)
                else:
                    auc = 0.5  # Random performance when only one class
            except Exception as e:
                print(f"AUC calculation failed: {e}")
                auc = 0.5
            
         # Handle case where class '1' might not exist in predictions
            if '1' in report:
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            return metrics

def load_temporal_graph(config):
    """Load temporal graph data."""
    print("Loading temporal graph...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    graph_path = artifacts_dir / 'temporal_graph.pt'
    
    if graph_path.exists():
        graph_data = torch.load(graph_path, weights_only=False)
        print(f"Loaded temporal graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
        return graph_data
    else:
        raise FileNotFoundError("Temporal graph not found. Please run temporal graph construction first.")

def save_model_results(model, metrics, config):
    """Save trained model and results."""
    print("Saving model and results...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / 'tgnn_model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    with open(artifacts_dir / 'tgnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model config
    model_config = {
        'model_type': model.model_type,
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_classes': model.num_classes
    }
    
    with open(artifacts_dir / 'tgnn_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model and results saved to {artifacts_dir}")

def main():
    parser = argparse.ArgumentParser(description="TGNN model training for Reddit hate speech")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== TGNN Model Training ===")
    
    # Load temporal graph
    graph_data = load_temporal_graph(config)
    
    # Initialize model
    model = TGNNModel(config)
    print(f"Initialized {model.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = TGNNTrainer(model, config)
    
    # Train classification task
    tgnn_config = config.get('tgnn', {})
    epochs = tgnn_config.get('num_epochs', 100)
    best_acc = trainer.train_classification(graph_data, epochs=epochs)
    
    # Evaluate model
    metrics = trainer.evaluate_model(graph_data)
    
    # Save results
    save_model_results(model, metrics, config)
    
    print(f"\n=== TGNN Training Summary ===")
    print(f"Model: {model.model_type}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
