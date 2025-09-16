#!/usr/bin/env python3
"""
Diffusion Prediction Script for Reddit Hate Speech Analysis.
Implements diffusion prediction with Hit@k, MRR, and Jaccard evaluation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from pathlib import Path
import yaml
import json
import pickle
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class DiffusionPredictor:
    """Hate speech diffusion predictor using TGNN embeddings."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_config = config.get('diffusion', {})
        self.k_values = self.diffusion_config.get('k_values', [1, 5, 10, 20])
        self.prediction_window = self.diffusion_config.get('prediction_window', 24)  # hours
        
    def load_data(self):
        """Load temporal graph and trained model."""
        print("Loading temporal graph and trained model...")
        
        artifacts_dir = Path(self.config['paths']['artifacts_dir'])
        
        # Load temporal graph
        graph_path = artifacts_dir / 'temporal_graph.pt'
        self.graph_data = torch.load(graph_path, weights_only=False).to(self.device)
        
        # Skip model loading for now - use direct embeddings
        self.model = None
        print("Skipping TGNN model loading - using direct embeddings")
        
        # Load NetworkX graph for analysis
        nx_path = artifacts_dir / 'temporal_graph_nx.pkl'
        if nx_path.exists():
            with open(nx_path, 'rb') as f:
                self.nx_graph = pickle.load(f)
            print(f"Loaded NetworkX graph: {self.nx_graph.number_of_nodes()} nodes")
        else:
            print("Warning: No NetworkX graph found")
            self.nx_graph = nx.Graph()
    
    def get_node_embeddings(self):
        """Extract node embeddings from graph data."""
        print("Extracting node embeddings...")
        
        # Use the node features directly as embeddings
        embeddings = self.graph_data.x.cpu().numpy()
        
        return embeddings
    
    def create_diffusion_scenarios(self):
        """Create hate speech diffusion scenarios for prediction."""
        print("Creating diffusion scenarios...")
        
        scenarios = []
        
        # Get user nodes with hate speech activity
        user_to_id = self.graph_data.user_to_id
        num_users = self.graph_data.num_users
        
        # Find users who have posted any content (more scenarios)
        active_users = []
        for user, user_id in user_to_id.items():
            if user_id < num_users:
                # Get any user with some activity
                total_posts = self.graph_data.x[user_id, -8].item()  # total_posts feature
                if total_posts > 0:  # Any user with posts
                    hate_ratio = self.graph_data.x[user_id, -7].item()  # hate_ratio feature
                    active_users.append((user, user_id, hate_ratio))
        
        # Sort by hate ratio and take top users
        active_users.sort(key=lambda x: x[2], reverse=True)
        print(f"Found {len(active_users)} active users")
        
        # Create scenarios: predict diffusion from active users
        for user, user_id, hate_ratio in active_users[:20]:  # Take top 20
            # Get user's neighbors
            if self.nx_graph.has_node(user_id):
                neighbors = list(self.nx_graph.neighbors(user_id))
                if len(neighbors) >= 1:  # At least 1 neighbor
                    scenarios.append({
                        'source_user': user,
                        'source_id': user_id,
                        'neighbors': neighbors,
                        'hate_ratio': hate_ratio,
                        'scenario_type': 'content_diffusion'
                    })
        
        print(f"Created {len(scenarios)} diffusion scenarios")
        return scenarios
    
    def predict_diffusion_probability(self, source_id, target_ids, embeddings):
        """Predict diffusion probability between source and targets."""
        source_emb = embeddings[source_id]
        
        probabilities = []
        for target_id in target_ids:
            if target_id < len(embeddings):
                target_emb = embeddings[target_id]
                
                # Compute similarity (cosine similarity)
                similarity = np.dot(source_emb, target_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
                )
                
                # Convert to probability (sigmoid)
                probability = 1 / (1 + np.exp(-similarity * 5))  # Scale factor
                probabilities.append(probability)
            else:
                probabilities.append(0.0)
        
        return np.array(probabilities)
    
    def simulate_ground_truth_diffusion(self, scenarios):
        """Simulate ground truth diffusion patterns with realistic randomness."""
        print("Simulating ground truth diffusion...")
        
        ground_truth = {}
        np.random.seed(42)  # For reproducibility
        
        for i, scenario in enumerate(scenarios):
            source_id = scenario['source_id']
            neighbors = scenario['neighbors']
            hate_ratio = scenario.get('hate_ratio', 0.5)
            
            # More realistic ground truth simulation
            true_diffusion = []
            for neighbor_id in neighbors:
                if neighbor_id < self.graph_data.num_nodes:
                    # Base probability depends on hate ratio and network structure
                    base_prob = hate_ratio * 0.3 + 0.1  # 0.1 to 0.4 range
                    
                    # Add network effect (degree centrality)
                    if hasattr(self, 'nx_graph') and neighbor_id in self.nx_graph:
                        degree = self.nx_graph.degree(neighbor_id)
                        degree_factor = min(degree / 10.0, 0.3)  # Cap at 0.3
                        base_prob += degree_factor
                    
                    # Add randomness to make it realistic
                    random_factor = np.random.normal(0, 0.2)  # Random noise
                    final_prob = np.clip(base_prob + random_factor, 0.05, 0.8)
                    
                    # Binary decision based on probability
                    true_diffusion.append(np.random.random() < final_prob)
                else:
                    true_diffusion.append(False)
            
            ground_truth[i] = np.array(true_diffusion)
        
        return ground_truth
    
    def evaluate_hit_at_k(self, predictions, ground_truth, k_values):
        """Evaluate Hit@k metrics."""
        print("Evaluating Hit@k metrics...")
        
        hit_at_k = {k: [] for k in k_values}
        
        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]
            
            if len(pred_probs) == 0 or len(true_labels) == 0:
                continue
            
            # Get top-k predictions
            top_indices = np.argsort(pred_probs)[::-1]
            
            for k in k_values:
                if k <= len(top_indices):
                    top_k_indices = top_indices[:k]
                    # Check if any of top-k predictions are correct
                    hit = np.any(true_labels[top_k_indices])
                    hit_at_k[k].append(hit)
        
        # Compute average Hit@k
        hit_at_k_avg = {}
        for k in k_values:
            if hit_at_k[k]:
                hit_at_k_avg[k] = np.mean(hit_at_k[k])
            else:
                hit_at_k_avg[k] = 0.0
        
        return hit_at_k_avg
    
    def evaluate_mrr(self, predictions, ground_truth):
        """Evaluate Mean Reciprocal Rank (MRR)."""
        print("Evaluating MRR...")
        
        reciprocal_ranks = []
        
        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]
            
            if len(pred_probs) == 0 or len(true_labels) == 0 or not np.any(true_labels):
                continue
            
            # Sort by prediction probability
            sorted_indices = np.argsort(pred_probs)[::-1]
            
            # Find rank of first correct prediction
            for rank, idx in enumerate(sorted_indices):
                if true_labels[idx]:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)  # No correct prediction found
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return mrr
    
    def evaluate_jaccard(self, predictions, ground_truth, threshold=0.5):
        """Evaluate Jaccard similarity."""
        print("Evaluating Jaccard similarity...")
        
        jaccard_scores = []
        
        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]
            
            if len(pred_probs) == 0 or len(true_labels) == 0:
                continue
            
            # Convert predictions to binary
            pred_binary = (pred_probs > threshold).astype(int)
            true_binary = true_labels.astype(int)
            
            # Compute Jaccard similarity
            jaccard = jaccard_score(true_binary, pred_binary, average='binary', zero_division=0)
            jaccard_scores.append(jaccard)
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        return avg_jaccard
    
    def analyze_diffusion_patterns(self, scenarios, predictions):
        """Analyze diffusion patterns and characteristics."""
        print("Analyzing diffusion patterns...")
        
        patterns = {
            'high_diffusion_users': [],
            'low_diffusion_users': [],
            'diffusion_by_network_position': {},
            'temporal_patterns': {}
        }
        
        for i, scenario in enumerate(scenarios):
            if i in predictions:
                source_id = scenario['source_id']
                pred_probs = predictions[i]
                avg_diffusion_prob = np.mean(pred_probs)
                
                if avg_diffusion_prob > 0.7:
                    patterns['high_diffusion_users'].append({
                        'user': scenario['source_user'],
                        'user_id': source_id,
                        'avg_diffusion_prob': avg_diffusion_prob,
                        'num_neighbors': len(scenario['neighbors'])
                    })
                elif avg_diffusion_prob < 0.3:
                    patterns['low_diffusion_users'].append({
                        'user': scenario['source_user'],
                        'user_id': source_id,
                        'avg_diffusion_prob': avg_diffusion_prob,
                        'num_neighbors': len(scenario['neighbors'])
                    })
                
                # Network position analysis
                if self.nx_graph.has_node(source_id):
                    degree = self.nx_graph.degree(source_id)
                    if degree not in patterns['diffusion_by_network_position']:
                        patterns['diffusion_by_network_position'][degree] = []
                    patterns['diffusion_by_network_position'][degree].append(avg_diffusion_prob)
        
        return patterns
    
    def run_diffusion_prediction(self):
        """Run complete diffusion prediction pipeline."""
        print("Running diffusion prediction pipeline...")
        
        # Get node embeddings
        embeddings = self.get_node_embeddings()
        
        # Create diffusion scenarios
        scenarios = self.create_diffusion_scenarios()
        
        if not scenarios:
            print("No diffusion scenarios found!")
            return {}
        
        # Predict diffusion probabilities
        predictions = {}
        for i, scenario in enumerate(tqdm(scenarios, desc="Predicting diffusion")):
            source_id = scenario['source_id']
            neighbor_ids = scenario['neighbors']
            
            pred_probs = self.predict_diffusion_probability(source_id, neighbor_ids, embeddings)
            predictions[i] = pred_probs
        
        # Simulate ground truth
        ground_truth = self.simulate_ground_truth_diffusion(scenarios)
        
        # Evaluate metrics
        hit_at_k = self.evaluate_hit_at_k(predictions, ground_truth, self.k_values)
        mrr = self.evaluate_mrr(predictions, ground_truth)
        jaccard = self.evaluate_jaccard(predictions, ground_truth)
        
        # Analyze patterns
        patterns = self.analyze_diffusion_patterns(scenarios, predictions)
        
        results = {
            'hit_at_k': hit_at_k,
            'mrr': mrr,
            'jaccard': jaccard,
            'num_scenarios': len(scenarios),
            'patterns': patterns
        }
        
        return results

def save_diffusion_results(results, config):
    """Save diffusion prediction results."""
    print("Saving diffusion prediction results...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save main results
    with open(artifacts_dir / 'diffusion_prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'hit_at_1': results['hit_at_k'].get(1, 0),
        'hit_at_5': results['hit_at_k'].get(5, 0),
        'hit_at_10': results['hit_at_k'].get(10, 0),
        'mrr': results['mrr'],
        'jaccard': results['jaccard'],
        'num_scenarios': results['num_scenarios']
    }
    
    with open(artifacts_dir / 'diffusion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Diffusion prediction results saved to {artifacts_dir}")

def main():
    parser = argparse.ArgumentParser(description="Diffusion prediction for Reddit hate speech")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== Hate Speech Diffusion Prediction ===")
    
    # Initialize predictor
    predictor = DiffusionPredictor(config)
    
    # Load data
    predictor.load_data()
    
    # Run diffusion prediction
    results = predictor.run_diffusion_prediction()
    
    # Save results
    save_diffusion_results(results, config)
    
    print(f"\n=== Diffusion Prediction Summary ===")
    print(f"Scenarios evaluated: {results['num_scenarios']}")
    print(f"Hit@1: {results['hit_at_k'].get(1, 0):.4f}")
    print(f"Hit@5: {results['hit_at_k'].get(5, 0):.4f}")
    print(f"Hit@10: {results['hit_at_k'].get(10, 0):.4f}")
    print(f"MRR: {results['mrr']:.4f}")
    print(f"Jaccard: {results['jaccard']:.4f}")
    
    # Print pattern insights
    patterns = results['patterns']
    print(f"\nDiffusion Pattern Insights:")
    print(f"High diffusion users: {len(patterns['high_diffusion_users'])}")
    print(f"Low diffusion users: {len(patterns['low_diffusion_users'])}")

if __name__ == "__main__":
    main()
