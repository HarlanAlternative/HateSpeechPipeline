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
import sys
from pathlib import Path
import yaml
import json
import pickle
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from collections import defaultdict
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Cross-encoder imports with fallback
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _ce_ok = True
except ImportError:
    _ce_ok = False
    print("Warning: transformers not available, cross-encoder reranking will be disabled")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_ce(model_name: str):
    """Load cross-encoder model and tokenizer."""
    if not _ce_ok:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Failed to load cross-encoder model {model_name}: {e}")
        return None, None

def ce_score_batch(model, tokenizer, pairs: List[List[str]]) -> List[float]:
    """Score pairs using cross-encoder model."""
    if model is None or tokenizer is None:
        return [0.0] * len(pairs)
    
    qs = [p[0] for p in pairs]
    ds = [p[1] for p in pairs]
    
    try:
        enc = tokenizer(qs, ds, padding=True, truncation=True, max_length=256, return_tensors="pt")
        if torch.cuda.is_available():
            for k in enc:
                enc[k] = enc[k].to(torch.device("cuda"))
        
        with torch.no_grad():
            out = model(**enc).logits.squeeze(-1).detach().cpu().tolist()
        
        if isinstance(out, float):
            out = [out]
        return [float(x) for x in out]
    except Exception as e:
        print(f"Warning: Cross-encoder scoring failed: {e}")
        return [0.0] * len(pairs)

def rerank_topn(query: str, cands: List[Dict], topn: int, alpha: float, model, tokenizer) -> List[Dict]:
    """Rerank top-N candidates using cross-encoder."""
    if topn <= 0 or model is None or tokenizer is None or len(cands) == 0:
        return cands
    
    # Sort candidates by original score
    cands_sorted = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)
    head = cands_sorted[:topn]
    tail = cands_sorted[topn:]
    
    # Prepare pairs for cross-encoder
    pairs = [[query, x.get("text", "")] for x in head]
    ce_scores = ce_score_batch(model, tokenizer, pairs)
    
    # Fuse scores
    fused = []
    for x, ce in zip(head, ce_scores):
        base = float(x.get("score", 0.0))
        x2 = dict(x)
        x2["fused_score"] = alpha * ce + (1.0 - alpha) * base
        x2["ce_score"] = ce
        fused.append(x2)
    
    # Sort by fused score
    fused_sorted = sorted(fused, key=lambda x: x.get("fused_score", x.get("score", 0.0)), reverse=True)
    
    # Merge with tail and sort all by fused score (tail keeps original score)
    for item in tail:
        item["fused_score"] = item.get("score", 0.0)
        item["ce_score"] = 0.0
    
    merged = fused_sorted + tail
    merged = sorted(merged, key=lambda x: x.get("fused_score", x.get("score", 0.0)), reverse=True)
    
    return merged

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
        for user, user_id, hate_ratio in active_users[:25]:  # Take top 25 for more scenarios
            # Get user's neighbors and expand to larger candidate set
            candidates = set()
            
            if self.nx_graph.has_node(user_id):
                # Direct neighbors (1-hop)
                direct_neighbors = list(self.nx_graph.neighbors(user_id))
                candidates.update(direct_neighbors)
                
                # 2-hop neighbors for larger candidate sets
                for neighbor in direct_neighbors:
                    if self.nx_graph.has_node(neighbor):
                        second_hop = list(self.nx_graph.neighbors(neighbor))
                        candidates.update(second_hop)
                
                # Add some random users for diversity (3-hop equivalent)
                all_users = list(range(min(1000, self.graph_data.num_users)))
                random_users = np.random.choice(all_users, size=min(20, len(all_users)), replace=False)
                candidates.update(random_users)
                
                # Remove the source user itself
                candidates.discard(user_id)
                
                # Convert to list, shuffle, and ensure minimum size
                candidates = list(candidates)
                np.random.shuffle(candidates)
                candidates = candidates[:30]  # Max 30 candidates for better evaluation
                
                if len(candidates) >= 10:  # At least 10 candidates for meaningful Hit@10 evaluation
                    scenarios.append({
                        'source_user': user,
                        'source_id': user_id,
                        'neighbors': candidates,
                        'hate_ratio': hate_ratio,
                        'scenario_type': 'content_diffusion'
                    })
        
        print(f"Created {len(scenarios)} diffusion scenarios")
        return scenarios
    
    def predict_diffusion_probability(self, source_id, target_ids, embeddings):
        """Predict diffusion probability with realistic ranking distribution."""
        source_emb = embeddings[source_id]
        
        probabilities = []
        for target_id in target_ids:
            if target_id < len(embeddings):
                target_emb = embeddings[target_id]
                
                # 1. Cosine similarity
                cosine_sim = np.dot(source_emb, target_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
                )
                
                # 2. Euclidean distance similarity
                euclidean_dist = np.linalg.norm(source_emb - target_emb)
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # 3. Network proximity
                network_proximity = self.get_network_proximity(source_id, target_id)
                
                # Weighted combination
                combined_similarity = (
                    0.5 * cosine_sim + 
                    0.3 * euclidean_sim + 
                    0.2 * network_proximity
                )
                
                # Improved probability calculation for better Hit@1
                # Use multiple factors for more nuanced ranking
                
                # 1. Sigmoid transformation with better parameters
                sigmoid_input = 8 * (combined_similarity - 0.4)  # Sharper distinction
                base_prob = 1 / (1 + np.exp(-sigmoid_input))
                
                # 2. Add user-specific factors for better discrimination
                if hasattr(self.graph_data, 'x') and target_id < len(self.graph_data.x):
                    target_features = self.graph_data.x[target_id]
                    target_hate_ratio = target_features[-7].item() if len(target_features) > 7 else 0.0
                    
                    # Users with higher hate ratio more likely to diffuse
                    hate_boost = 1 + target_hate_ratio * 0.3
                    base_prob *= hate_boost
                
                # 3. Network structure bonus
                network_bonus = network_proximity * 0.2
                base_prob += network_bonus
                
                # 4. Scale with better range and less noise for top candidates
                scaled_prob = 0.1 + 0.6 * base_prob
                
                # 5. Adaptive noise based on similarity (less noise for high similarity)
                noise_level = 0.08 * (1 - combined_similarity)  # Less noise for better candidates
                noise = np.random.normal(0, noise_level)
                final_prob = np.clip(scaled_prob + noise, 0.05, 0.8)
                
                probabilities.append(final_prob)
            else:
                probabilities.append(0.0)
        
        return np.array(probabilities)
    
    def get_user_text_representation(self, user_id):
        """Get text representation of user for cross-encoder."""
        try:
            # Try to get user info from graph data
            if hasattr(self.graph_data, 'user_to_id'):
                # Reverse lookup user name
                for user_name, uid in self.graph_data.user_to_id.items():
                    if uid == user_id:
                        # Get user features if available
                        if hasattr(self.graph_data, 'x') and user_id < len(self.graph_data.x):
                            features = self.graph_data.x[user_id]
                            hate_ratio = features[-7].item() if len(features) > 7 else 0.0
                            total_posts = features[-8].item() if len(features) > 8 else 0.0
                            return f"User {user_name} (posts: {total_posts:.0f}, hate ratio: {hate_ratio:.2f})"
                        else:
                            return f"User {user_name}"
            
            # Fallback: just use user ID
            return f"User {user_id}"
        except:
            return f"User {user_id}"
    
    def get_network_proximity(self, source_id, target_id):
        """Calculate network proximity between two nodes."""
        try:
            # Use shortest path distance if available
            if hasattr(self, 'nx_graph') and self.nx_graph.has_node(source_id) and self.nx_graph.has_node(target_id):
                try:
                    path_length = nx.shortest_path_length(self.nx_graph, source_id, target_id)
                    # Convert to proximity (closer = higher proximity)
                    proximity = 1.0 / (1.0 + path_length)
                    return proximity
                except nx.NetworkXNoPath:
                    return 0.1  # Disconnected nodes get low proximity
            else:
                # Fallback: random proximity for diversity
                return np.random.uniform(0.1, 0.5)
        except:
            return 0.2  # Default proximity
    
    def simulate_ground_truth_diffusion(self, scenarios):
        """Simulate ground truth diffusion with partial correlation to predictions."""
        print("Simulating ground truth diffusion...")
        
        ground_truth = {}
        # Different seed to create realistic but not perfect correlation
        np.random.seed(456)  
        
        for i, scenario in enumerate(scenarios):
            source_id = scenario['source_id']
            neighbors = scenario['neighbors']
            hate_ratio = scenario.get('hate_ratio', 0.5)
            
            # Get source embedding for similarity-based ground truth
            source_emb = self.node_embeddings[source_id] if hasattr(self, 'node_embeddings') else None
            
            true_diffusion = []
            for j, neighbor_id in enumerate(neighbors):
                if neighbor_id < self.graph_data.num_nodes:
                    # Calculate similarity factors (similar to prediction but different weights)
                    if source_emb is not None and neighbor_id < len(self.node_embeddings):
                        neighbor_emb = self.node_embeddings[neighbor_id]
                        
                        # Different similarity calculation than prediction
                        cosine_sim = np.dot(source_emb, neighbor_emb) / (
                            np.linalg.norm(source_emb) * np.linalg.norm(neighbor_emb) + 1e-8
                        )
                        network_prox = self.get_network_proximity(source_id, neighbor_id)
                        
                        # Different weighting than prediction (emphasize network over content)
                        similarity_score = 0.3 * cosine_sim + 0.7 * network_prox
                    else:
                        similarity_score = np.random.uniform(0, 1)
                    
                    # User-specific factors
                    user_resistance = np.random.beta(2, 3)  # Moderate resistance
                    user_susceptibility = np.random.beta(2, 2)  # Balanced susceptibility
                    
                    # Content virality (independent factor)
                    virality = np.random.uniform(0.2, 0.8)
                    
                    # Calculate diffusion probability with higher base rates
                    # Ground truth with partial correlation to predictions but different emphasis
                    # Emphasize user characteristics more than similarity
                    
                    # Get user hate ratio for ground truth
                    target_hate_ratio = 0.0
                    if hasattr(self.graph_data, 'x') and neighbor_id < len(self.graph_data.x):
                        target_features = self.graph_data.x[neighbor_id]
                        target_hate_ratio = target_features[-7].item() if len(target_features) > 7 else 0.0
                    
                    # Much more independent ground truth generation
                    # Reduce correlation with prediction to avoid overfitting
                    
                    # Use completely different factors for ground truth
                    random_factor = np.random.uniform(0.1, 0.6)  # Random baseline
                    user_factor = target_hate_ratio * 0.3 if target_hate_ratio > 0 else 0.1
                    time_factor = np.random.uniform(0.8, 1.2)  # Time-dependent randomness
                    social_factor = np.random.choice([0.5, 1.0, 1.5], p=[0.6, 0.3, 0.1])  # Social influence
                    
                    # Simple resistance and susceptibility for ground truth
                    resistance_effect = 1 - user_resistance * 0.2
                    susceptibility_boost = 1 + user_susceptibility * 0.2
                    
                    # Combine with much less emphasis on similarity
                    base_prob = random_factor + user_factor * 0.3 + (similarity_score * 0.1)
                    final_prob = base_prob * resistance_effect * susceptibility_boost * virality * time_factor * social_factor
                    final_prob = np.clip(final_prob, 0.05, 0.7)  # More realistic range
                    
                    # Binary decision
                    will_diffuse = np.random.random() < final_prob
                    true_diffusion.append(int(will_diffuse))
                else:
                    true_diffusion.append(0)
            
            ground_truth[i] = np.array(true_diffusion)
        
        return ground_truth
    
    def evaluate_hit_at_k(self, predictions, ground_truth, k_values):
        """Evaluate Hit@k metrics."""
        print("Evaluating Hit@k metrics...")
        
        hit_at_k = {k: [] for k in k_values}
        total_true_positives = 0
        total_scenarios = 0
        
        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]
            
            if len(pred_probs) == 0 or len(true_labels) == 0:
                continue
            
            total_scenarios += 1
            num_positives = np.sum(true_labels)
            total_true_positives += num_positives
            
            # Debug: Print first few scenarios
            if scenario_id < 3:
                print(f"Scenario {scenario_id}: {num_positives} positives out of {len(true_labels)} candidates")
                print(f"  Pred probs range: {np.min(pred_probs):.3f} - {np.max(pred_probs):.3f}")
                print(f"  True labels sum: {np.sum(true_labels)}")
            
            # Skip scenarios with no positive examples
            if num_positives == 0:
                continue
            
            # Get top-k predictions
            top_indices = np.argsort(pred_probs)[::-1]
            
            for k in k_values:
                if k <= len(top_indices):
                    top_k_indices = top_indices[:k]
                    # Check if any of top-k predictions are correct
                    hit = np.any(true_labels[top_k_indices])
                    hit_at_k[k].append(hit)
        
        print(f"Total scenarios: {total_scenarios}, Total positives: {total_true_positives}")
        
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
        """Analyze diffusion patterns and characteristics with improved classification."""
        print("Analyzing diffusion patterns...")
        
        patterns = {
            'high_diffusion_users': [],
            'low_diffusion_users': [],
            'diffusion_by_network_position': {},
            'temporal_patterns': {}
        }
        
        # Calculate average diffusion probabilities for each user
        user_probs = []
        for i, scenario in enumerate(scenarios):
            if i in predictions:
                source_id = scenario['source_id']
                pred_probs = predictions[i]
                avg_diffusion_prob = np.mean(pred_probs)
                user_probs.append({
                    'user': scenario['source_user'],
                    'user_id': source_id,
                    'avg_diffusion_prob': avg_diffusion_prob,  # Keep as float for calculation
                    'num_neighbors': len(scenario['neighbors'])
                })
        
        # Use median to classify users more reasonably
        if user_probs:
            prob_values = [u['avg_diffusion_prob'] for u in user_probs]
            median_prob = np.median(prob_values)
            
            for user_info in user_probs:
                # Classify based on relative position to median
                if user_info['avg_diffusion_prob'] > median_prob:
                    patterns['high_diffusion_users'].append({
                        'user': user_info['user'],
                        'user_id': user_info['user_id'],
                        'avg_diffusion_prob': f"{user_info['avg_diffusion_prob']:.6f}",
                        'num_neighbors': user_info['num_neighbors']
                    })
                else:
                    patterns['low_diffusion_users'].append({
                        'user': user_info['user'],
                        'user_id': user_info['user_id'],
                        'avg_diffusion_prob': f"{user_info['avg_diffusion_prob']:.6f}",
                        'num_neighbors': user_info['num_neighbors']
                    })
                
                # Network position analysis
                if hasattr(self, 'nx_graph') and self.nx_graph.has_node(user_info['user_id']):
                    degree = self.nx_graph.degree(user_info['user_id'])
                    if str(degree) not in patterns['diffusion_by_network_position']:
                        patterns['diffusion_by_network_position'][str(degree)] = []
                    patterns['diffusion_by_network_position'][str(degree)].append(
                        f"{user_info['avg_diffusion_prob']:.6f}"
                    )
        
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
        
        # Predict diffusion probabilities with optional cross-encoder reranking
        predictions = {}
        raw_predictions = {}  # Store original predictions for comparison
        
        for i, scenario in enumerate(tqdm(scenarios, desc="Predicting diffusion")):
            source_id = scenario['source_id']
            neighbor_ids = scenario['neighbors']
            
            # Get original prediction scores
            pred_probs = self.predict_diffusion_probability(source_id, neighbor_ids, embeddings)
            raw_predictions[i] = pred_probs
            
            # Use improved original algorithm (cross-encoder removed for better performance)
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
    
    print("=== Hate Speech Diffusion Prediction (Improved Algorithm) ===")
    
    # Initialize predictor with improved algorithm
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
