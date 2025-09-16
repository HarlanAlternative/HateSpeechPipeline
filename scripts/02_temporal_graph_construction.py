#!/usr/bin/env python3
"""
Temporal Graph Construction Script for Reddit Hate Speech Analysis.
Creates temporal graphs with user/post nodes and time-aware edges for TGNN models.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, TemporalData
import argparse
from pathlib import Path
import yaml
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TemporalGraphBuilder:
    """Builder for temporal graphs from Reddit data."""
    
    def __init__(self, config):
        self.config = config
        self.user_to_id = {}
        self.post_to_id = {}
        self.node_features = []
        self.edges = []
        self.edge_features = []
        self.timestamps = []
        
    def load_data(self):
        """Load prepared data and features."""
        print("Loading prepared data...")
        
        artifacts_dir = Path(self.config['paths']['artifacts_dir'])
        
        # Load enhanced dataset with BERT features
        bert_path = artifacts_dir / 'bert_enhanced_dataset.parquet'
        if bert_path.exists():
            self.df = pd.read_parquet(bert_path)
            print(f"Loaded BERT-enhanced dataset: {len(self.df)} samples")
        else:
            # Fallback to balanced dataset
            balanced_path = artifacts_dir / 'balanced_dataset.parquet'
            self.df = pd.read_parquet(balanced_path)
            print(f"Loaded balanced dataset: {len(self.df)} samples")
        
        # Load BERT embeddings
        embedding_path = artifacts_dir / 'bert_embeddings.npy'
        if embedding_path.exists():
            self.embeddings = np.load(embedding_path)
            print(f"Loaded embeddings: {self.embeddings.shape}")
        else:
            print("No embeddings found, using dummy embeddings")
            self.embeddings = np.random.randn(len(self.df), 768)  # Dummy embeddings
        
        # Load user features
        user_features_path = artifacts_dir / 'user_features.json'
        if user_features_path.exists():
            with open(user_features_path, 'r') as f:
                self.user_features = json.load(f)
            print(f"Loaded user features for {len(self.user_features)} users")
        else:
            self.user_features = {}
            print("No user features found")
    
    def create_node_mappings(self):
        """Create mappings from users/posts to node IDs."""
        print("Creating node mappings...")
        
        # Create user nodes
        unique_users = self.df['author'].dropna().unique()
        self.user_to_id = {user: i for i, user in enumerate(unique_users)}
        
        # Create post nodes (offset by number of users)
        unique_posts = self.df['id'].dropna().unique()
        offset = len(self.user_to_id)
        self.post_to_id = {post: i + offset for i, post in enumerate(unique_posts)}
        
        print(f"Created {len(self.user_to_id)} user nodes and {len(self.post_to_id)} post nodes")
        
        return len(self.user_to_id) + len(self.post_to_id)
    
    def create_node_features(self, total_nodes):
        """Create node feature matrix."""
        print("Creating node features...")
        
        # Initialize feature matrix
        feature_dim = self.embeddings.shape[1] + 8  # BERT + user features
        node_features = np.zeros((total_nodes, feature_dim))
        
        # Process user nodes
        for user, user_id in tqdm(self.user_to_id.items(), desc="Processing user nodes"):
            # Get user data
            user_data = self.df[self.df['author'] == user]
            
            if len(user_data) > 0:
                # Use first post's embedding as user representation
                first_idx = user_data.index[0]
                if first_idx < len(self.embeddings):
                    node_features[user_id, :self.embeddings.shape[1]] = self.embeddings[first_idx]
                
                # Add user-specific features
                if user in self.user_features:
                    uf = self.user_features[user]
                    base_idx = self.embeddings.shape[1]
                    node_features[user_id, base_idx:base_idx+8] = [
                        uf.get('total_posts', 0),
                        uf.get('hate_ratio', 0),
                        uf.get('offensive_ratio', 0),
                        uf.get('subreddit_diversity', 0),
                        uf.get('avg_posting_interval_hours', 0),
                        uf.get('avg_text_length', 0),
                        1.0,  # is_user flag
                        0.0   # is_post flag
                    ]
        
        # Process post nodes
        for post, post_id in tqdm(self.post_to_id.items(), desc="Processing post nodes"):
            # Get post data
            post_data = self.df[self.df['id'] == post]
            
            if len(post_data) > 0:
                post_idx = post_data.index[0]
                if post_idx < len(self.embeddings):
                    node_features[post_id, :self.embeddings.shape[1]] = self.embeddings[post_idx]
                
                # Add post-specific features
                base_idx = self.embeddings.shape[1]
                post_row = post_data.iloc[0]
                node_features[post_id, base_idx:base_idx+8] = [
                    1.0,  # post count (always 1 for posts)
                    post_row.get('davidson_label', 0) == 2,  # is_hate
                    post_row.get('davidson_label', 0) == 1,  # is_offensive
                    1.0,  # subreddit diversity (1 for single post)
                    0.0,  # posting interval (N/A for posts)
                    len(post_row.get('text_content', '')),  # text length
                    0.0,  # is_user flag
                    1.0   # is_post flag
                ]
        
        self.node_features = torch.FloatTensor(node_features)
        print(f"Created node features: {self.node_features.shape}")
    
    def create_temporal_edges(self):
        """Create temporal edges between users and posts."""
        print("Creating temporal edges...")
        
        edges = []
        edge_features = []
        timestamps = []
        
        # Create user-post edges (authorship)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating authorship edges"):
            user = row['author']
            post = row['id']
            
            if pd.notna(user) and pd.notna(post):
                if user in self.user_to_id and post in self.post_to_id:
                    user_id = self.user_to_id[user]
                    post_id = self.post_to_id[post]
                    
                    # Bidirectional edges
                    edges.append([user_id, post_id])
                    edges.append([post_id, user_id])
                    
                    # Edge features: [edge_type, hate_label, subreddit_hash]
                    subreddit_hash = hash(row.get('subreddit', '')) % 1000  # Simple subreddit encoding
                    edge_feat = [
                        1.0,  # authorship edge type
                        float(row.get('davidson_label', 0) == 2),  # hate label
                        subreddit_hash / 1000.0  # normalized subreddit hash
                    ]
                    edge_features.extend([edge_feat, edge_feat])  # Same features for both directions
                    
                    # Timestamps
                    timestamp = row.get('created_utc', 0)
                    timestamps.extend([timestamp, timestamp])
        
        # Create user-user edges (co-commenting in same thread)
        print("Creating co-comment edges...")
        thread_groups = self.df[self.df['source'] == 'comment'].groupby('link_id')
        
        for link_id, group in tqdm(thread_groups, desc="Processing comment threads"):
            if pd.isna(link_id):
                continue
            
            authors = group['author'].dropna().unique()
            authors = [a for a in authors if a in self.user_to_id]
            
            # Create edges between co-commenters (limit to avoid explosion)
            if len(authors) > 1:
                for i, author1 in enumerate(authors[:10]):  # Limit to first 10 authors
                    for author2 in authors[i+1:min(i+6, len(authors))]:  # Connect to next 5
                        user1_id = self.user_to_id[author1]
                        user2_id = self.user_to_id[author2]
                        
                        # Bidirectional co-comment edges
                        edges.append([user1_id, user2_id])
                        edges.append([user2_id, user1_id])
                        
                        # Edge features for co-comment
                        avg_timestamp = group['created_utc'].mean()
                        edge_feat = [
                            2.0,  # co-comment edge type
                            0.0,  # not directly hate-related
                            hash(link_id) % 1000 / 1000.0  # thread identifier
                        ]
                        edge_features.extend([edge_feat, edge_feat])
                        timestamps.extend([avg_timestamp, avg_timestamp])
        
        self.edges = torch.LongTensor(edges).t()
        self.edge_features = torch.FloatTensor(edge_features)
        self.timestamps = torch.FloatTensor(timestamps)
        
        print(f"Created {self.edges.shape[1]} temporal edges")
    
    def build_temporal_graph(self):
        """Build the complete temporal graph."""
        print("Building temporal graph...")
        
        # Create the temporal graph data
        graph_data = TemporalData(
            x=self.node_features,
            edge_index=self.edges,
            edge_attr=self.edge_features,
            t=self.timestamps,
            num_nodes=self.node_features.shape[0]
        )
        
        # Add metadata
        graph_data.user_to_id = self.user_to_id
        graph_data.post_to_id = self.post_to_id
        graph_data.num_users = len(self.user_to_id)
        graph_data.num_posts = len(self.post_to_id)
        
        return graph_data
    
    def create_networkx_graph(self):
        """Create NetworkX graph for analysis and visualization."""
        print("Creating NetworkX graph...")
        
        G = nx.Graph()
        
        # Add nodes
        for user, user_id in self.user_to_id.items():
            G.add_node(user_id, type='user', name=user)
        
        for post, post_id in self.post_to_id.items():
            G.add_node(post_id, type='post', name=post)
        
        # Add edges (sample for efficiency)
        edge_sample_size = min(1000, self.edges.shape[1])
        edge_indices = np.random.choice(self.edges.shape[1], edge_sample_size, replace=False)
        
        for i in edge_indices:
            src, dst = self.edges[:, i].tolist()
            timestamp = self.timestamps[i].item()
            edge_type = self.edge_features[i, 0].item()
            
            G.add_edge(src, dst, timestamp=timestamp, edge_type=edge_type)
        
        return G

def save_temporal_graph(graph_data, nx_graph, config):
    """Save temporal graph data."""
    print("Saving temporal graph...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save PyTorch Geometric temporal data
    torch.save(graph_data, artifacts_dir / 'temporal_graph.pt')
    
    # Save NetworkX graph
    with open(artifacts_dir / 'temporal_graph_nx.pkl', 'wb') as f:
        pickle.dump(nx_graph, f)
    
    # Save graph statistics
    stats = {
        'num_nodes': graph_data.num_nodes,
        'num_edges': graph_data.edge_index.shape[1],
        'num_users': graph_data.num_users,
        'num_posts': graph_data.num_posts,
        'feature_dim': graph_data.x.shape[1],
        'edge_feature_dim': graph_data.edge_attr.shape[1],
        'time_span': float(graph_data.t.max() - graph_data.t.min()),
        'avg_degree': float(graph_data.edge_index.shape[1] / graph_data.num_nodes)
    }
    
    with open(artifacts_dir / 'temporal_graph_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Temporal graph saved:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Users: {stats['num_users']}")
    print(f"  Posts: {stats['num_posts']}")

def main():
    parser = argparse.ArgumentParser(description="Temporal graph construction for Reddit data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== Temporal Graph Construction ===")
    
    # Initialize graph builder
    builder = TemporalGraphBuilder(config)
    
    # Load data
    builder.load_data()
    
    # Create node mappings
    total_nodes = builder.create_node_mappings()
    
    # Create node features
    builder.create_node_features(total_nodes)
    
    # Create temporal edges
    builder.create_temporal_edges()
    
    # Build temporal graph
    graph_data = builder.build_temporal_graph()
    
    # Create NetworkX graph
    nx_graph = builder.create_networkx_graph()
    
    # Save results
    save_temporal_graph(graph_data, nx_graph, config)
    
    print(f"\n=== Temporal Graph Construction Summary ===")
    print(f"Total nodes: {graph_data.num_nodes}")
    print(f"Total edges: {graph_data.edge_index.shape[1]}")
    print(f"Node feature dimension: {graph_data.x.shape[1]}")
    print(f"Edge feature dimension: {graph_data.edge_attr.shape[1]}")
    print(f"Time span: {graph_data.t.max() - graph_data.t.min():.0f} seconds")

if __name__ == "__main__":
    main()
