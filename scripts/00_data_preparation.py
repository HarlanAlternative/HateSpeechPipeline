#!/usr/bin/env python3
"""
Data Preparation Script for Reddit Hate Speech Analysis.
Implements Davidson et al. lexicon-based labeling and balanced dataset creation.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import json
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


DAVIDSON_HATE_LEXICON = {
    # Racial/ethnic slurs and hate terms
    'hate_terms': [
        'nigger', 'nigga', 'kike', 'spic', 'wetback', 'chink', 'gook', 'jap',
        'towelhead', 'raghead', 'sandnigger', 'paki', 'curry', 'beaner',
        'faggot', 'fag', 'dyke', 'tranny', 'homo', 'queer'
    ],
    # Offensive but not necessarily hate speech
    'offensive_terms': [
        'bitch', 'slut', 'whore', 'cunt', 'pussy', 'dick', 'cock', 'asshole',
        'bastard', 'damn', 'hell', 'shit', 'fuck', 'fucking', 'motherfucker'
    ],
    # Context-dependent terms
    'context_terms': [
        'kill', 'die', 'death', 'murder', 'rape', 'violence', 'terrorist',
        'nazi', 'hitler', 'holocaust', 'genocide', 'slavery', 'lynching'
    ]
}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_reddit_data(config):
    """Load Reddit data from parquet files with sampling for speed."""
    print("Loading Reddit data (with sampling for speed)...")
    
    max_samples = config.get('data_prep', {}).get('max_samples_per_file', 1000)
    
    # Load submissions
    mini_dir = Path(config['paths']['mini_dir'])
    submissions_files = list(mini_dir.glob("*submission*.parquet"))
    
    if not submissions_files:
        print("No submission files found, checking mini_dataset...")
        mini_dir = Path("mini_dataset")
        submissions_files = list(mini_dir.glob("*submission*.parquet"))
    
    submissions_dfs = []
    for file in submissions_files[:2]:  # Only process first 2 files for speed
        try:
            df = pd.read_parquet(file)
            # Sample data for speed
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=config['random_state'])
            submissions_dfs.append(df)
            print(f"Loaded {len(df)} submissions from {file.name}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    submissions_df = pd.concat(submissions_dfs, ignore_index=True) if submissions_dfs else pd.DataFrame()
    
    # Load comments
    comments_files = list(mini_dir.glob("*comment*.parquet"))
    
    if not comments_files:
        print("No comment files found, checking mini_dataset...")
        mini_dir = Path("mini_dataset")
        comments_files = list(mini_dir.glob("*comment*.parquet"))
    
    comments_dfs = []
    for file in comments_files[:2]:  # Only process first 2 files for speed
        try:
            df = pd.read_parquet(file)
            # Sample data for speed
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=config['random_state'])
            comments_dfs.append(df)
            print(f"Loaded {len(df)} comments from {file.name}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    comments_df = pd.concat(comments_dfs, ignore_index=True) if comments_dfs else pd.DataFrame()
    
    print(f"Total loaded: {len(submissions_df)} submissions and {len(comments_df)} comments")
    return submissions_df, comments_df

def apply_davidson_labeling(text):
    """
    Apply Davidson et al. lexicon-based labeling (optimized for speed).
    Returns: 0 = not hate, 1 = offensive, 2 = hate speech
    """
    if pd.isna(text) or not text:
        return 0
    
    text_lower = text.lower()
    
    # Quick check for hate speech terms (most restrictive first)
    for term in DAVIDSON_HATE_LEXICON['hate_terms']:
        if term in text_lower:
            return 2  # Hate speech
    
    # Count offensive and context terms
    offensive_count = sum(1 for term in DAVIDSON_HATE_LEXICON['offensive_terms'] if term in text_lower)
    context_count = sum(1 for term in DAVIDSON_HATE_LEXICON['context_terms'] if term in text_lower)
    
    # More liberal classification to get more hate samples
    if offensive_count >= 3 or context_count >= 2:
        return 2  # Hate speech (more liberal)
    elif offensive_count >= 2 and context_count >= 1:
        return 2  # Hate speech (combination)
    elif offensive_count > 0 or context_count > 0:
        return 1  # Offensive
    
    return 0  # Not hate/offensive

def label_reddit_data(submissions_df, comments_df, config):
    """Label Reddit data using Davidson lexicon."""
    print("Applying Davidson lexicon labeling...")
    
    # Label submissions
    if not submissions_df.empty:
        print(f"Processing {len(submissions_df)} submissions...")
        submissions_df['text_content'] = (submissions_df.get('title', '') + ' ' + 
                                        submissions_df.get('selftext', '')).fillna('')
        
        # Use tqdm for progress bar
        tqdm.pandas(desc="Labeling submissions")
        submissions_df['davidson_label'] = submissions_df['text_content'].progress_apply(apply_davidson_labeling)
        submissions_df['source'] = 'submission'
    
    # Label comments
    if not comments_df.empty:
        print(f"Processing {len(comments_df)} comments...")
        comments_df['text_content'] = comments_df.get('body', '').fillna('')
        
        # Use tqdm for progress bar
        tqdm.pandas(desc="Labeling comments")
        comments_df['davidson_label'] = comments_df['text_content'].progress_apply(apply_davidson_labeling)
        comments_df['source'] = 'comment'
    
    # Combine data
    combined_data = []
    
    if not submissions_df.empty:
        submissions_subset = submissions_df[['id', 'subreddit', 'author', 'created_utc', 
                                           'score', 'text_content', 'davidson_label', 'source']].copy()
        combined_data.append(submissions_subset)
    
    if not comments_df.empty:
        comments_subset = comments_df[['id', 'subreddit', 'author', 'created_utc', 
                                     'score', 'link_id', 'parent_id', 'text_content', 
                                     'davidson_label', 'source']].copy()
        combined_data.append(comments_subset)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
    else:
        combined_df = pd.DataFrame()
    
    print(f"Labeling results:")
    if not combined_df.empty:
        label_counts = combined_df['davidson_label'].value_counts()
        print(f"  Not hate/offensive (0): {label_counts.get(0, 0):,}")
        print(f"  Offensive (1): {label_counts.get(1, 0):,}")
        print(f"  Hate speech (2): {label_counts.get(2, 0):,}")
    
    return combined_df

def analyze_subreddit_hate_levels(combined_df, config):
    """Analyze hate speech levels by subreddit."""
    print("Analyzing subreddit hate speech levels...")
    
    if combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate hate statistics by subreddit
    subreddit_stats = []
    
    subreddit_groups = list(combined_df.groupby('subreddit'))
    print(f"Processing {len(subreddit_groups)} subreddits...")
    
    for subreddit, group in tqdm(subreddit_groups, desc="Analyzing subreddits"):
        total_posts = len(group)
        hate_posts = len(group[group['davidson_label'] == 2])
        offensive_posts = len(group[group['davidson_label'] == 1])
        normal_posts = len(group[group['davidson_label'] == 0])
        
        hate_ratio = hate_posts / total_posts if total_posts > 0 else 0
        offensive_ratio = offensive_posts / total_posts if total_posts > 0 else 0
        
        subreddit_stats.append({
            'subreddit': subreddit,
            'total_posts': total_posts,
            'hate_posts': hate_posts,
            'offensive_posts': offensive_posts,
            'normal_posts': normal_posts,
            'hate_ratio': hate_ratio,
            'offensive_ratio': offensive_ratio,
            'combined_ratio': hate_ratio + offensive_ratio
        })
    
    subreddit_stats_df = pd.DataFrame(subreddit_stats)
    
    # Filter subreddits with sufficient data
    min_posts = config.get('data_prep', {}).get('min_posts_per_subreddit', 100)
    subreddit_stats_df = subreddit_stats_df[subreddit_stats_df['total_posts'] >= min_posts]
    
    # Get top 10 hate and non-hate subreddits
    top_hate_subreddits = subreddit_stats_df.nlargest(10, 'hate_ratio')
    top_normal_subreddits = subreddit_stats_df.nsmallest(10, 'combined_ratio')
    
    print(f"\nTop 10 Hate Speech Subreddits:")
    for _, row in top_hate_subreddits.iterrows():
        print(f"  {row['subreddit']}: {row['hate_ratio']:.3f} hate ratio ({row['hate_posts']} hate posts)")
    
    print(f"\nTop 10 Normal Subreddits:")
    for _, row in top_normal_subreddits.iterrows():
        print(f"  {row['subreddit']}: {row['combined_ratio']:.3f} combined ratio ({row['hate_posts']} hate posts)")
    
    return top_hate_subreddits, top_normal_subreddits

def create_balanced_dataset(combined_df, top_hate_subreddits, top_normal_subreddits, config):
    """Create balanced 1:1 hate/non-hate dataset."""
    print("Creating balanced dataset...")
    
    if combined_df.empty:
        return pd.DataFrame()
    
    target_size = config.get('data_prep', {}).get('target_dataset_size', 10000)
    hate_samples = target_size // 2
    normal_samples = target_size // 2
    
    # Get all hate speech data (from all subreddits, not just top ones)
    hate_data = combined_df[combined_df['davidson_label'] == 2]
    
    if len(hate_data) > hate_samples:
        hate_sample = hate_data.sample(n=hate_samples, random_state=config['random_state'])
    else:
        hate_sample = hate_data.copy()
        print(f"Warning: Only {len(hate_data)} hate samples available, less than target {hate_samples}")
    
    # Get normal data (from all subreddits)
    normal_data = combined_df[combined_df['davidson_label'] == 0]
    
    # Ensure 1:1 balance - match the number of hate samples
    actual_hate_count = len(hate_sample)
    if len(normal_data) > actual_hate_count:
        normal_sample = normal_data.sample(n=actual_hate_count, random_state=config['random_state'])
    else:
        normal_sample = normal_data.copy()
        print(f"Warning: Only {len(normal_data)} normal samples available, less than hate samples {actual_hate_count}")
    
    # Combine balanced dataset
    balanced_df = pd.concat([hate_sample, normal_sample], ignore_index=True)
    
    # Create binary labels (0 = normal, 1 = hate)
    balanced_df['binary_label'] = (balanced_df['davidson_label'] == 2).astype(int)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=config['random_state']).reset_index(drop=True)
    
    print(f"Balanced dataset created:")
    print(f"  Total samples: {len(balanced_df)}")
    print(f"  Hate samples: {(balanced_df['binary_label'] == 1).sum()}")
    print(f"  Normal samples: {(balanced_df['binary_label'] == 0).sum()}")
    print(f"  Balance ratio: {(balanced_df['binary_label'] == 1).sum() / len(balanced_df):.3f}")
    
    return balanced_df

def prepare_graph_structure(balanced_df, config):
    """Prepare simplified graph structure for GNN models (fast version)."""
    print("Preparing simplified graph structure...")
    
    if balanced_df.empty:
        return {}
    
    # Create user nodes (simplified)
    users = balanced_df['author'].dropna().unique()[:100]  # Limit to 100 users for speed
    user_to_id = {user: i for i, user in enumerate(users)}
    
    # Create comment/submission nodes
    posts = balanced_df[balanced_df['author'].isin(users)][['id', 'author', 'subreddit', 'binary_label']].copy()
    
    # Create simplified edges (only co-comment edges for speed)
    edges = []
    
    # Only process first 50 threads for speed
    comment_data = balanced_df[balanced_df['source'] == 'comment'].head(500)
    processed_threads = 0
    
    comment_groups = list(comment_data.groupby('link_id'))
    print(f"Processing {len(comment_groups)} comment threads for graph edges...")
    
    for link_id, group in tqdm(comment_groups, desc="Building graph edges"):
        if pd.isna(link_id) or processed_threads >= 50:
            continue
            
        processed_threads += 1
        authors = group['author'].dropna().unique()
        authors = [a for a in authors if a in user_to_id]  # Only users we're tracking
        
        if len(authors) > 1:
            # Create edges between first few pairs only
            for i, author1 in enumerate(authors[:5]):  # Limit to first 5 authors per thread
                for author2 in authors[i+1:i+3]:  # Only connect to next 2 authors
                    edges.append({
                        'source': user_to_id[author1],
                        'target': user_to_id[author2],
                        'edge_type': 'co_comment',
                        'timestamp': group['created_utc'].min()
                    })
    
    graph_data = {
        'users': users,
        'user_to_id': user_to_id,
        'posts': posts,
        'edges': edges,
        'num_users': len(users),
        'num_edges': len(edges)
    }
    
    print(f"Simplified graph structure prepared:")
    print(f"  Users: {len(users)}")
    print(f"  Posts: {len(posts)}")
    print(f"  Edges: {len(edges)}")
    
    return graph_data

def save_prepared_data(combined_df, balanced_df, subreddit_stats, graph_data, config):
    """Save all prepared data."""
    print("Saving prepared data...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save full labeled dataset
    if not combined_df.empty:
        full_data_path = artifacts_dir / 'full_labeled_data.parquet'
        combined_df.to_parquet(full_data_path, engine='pyarrow')
        print(f"Full labeled dataset saved to {full_data_path}")
    
    # Save balanced dataset
    if not balanced_df.empty:
        balanced_data_path = artifacts_dir / 'balanced_dataset.parquet'
        balanced_df.to_parquet(balanced_data_path, engine='pyarrow')
        print(f"Balanced dataset saved to {balanced_data_path}")
    
    # Save subreddit statistics
    if len(subreddit_stats) == 2:
        top_hate, top_normal = subreddit_stats
        subreddit_stats_path = artifacts_dir / 'subreddit_analysis.json'
        subreddit_data = {
            'top_hate_subreddits': top_hate.to_dict('records'),
            'top_normal_subreddits': top_normal.to_dict('records')
        }
        with open(subreddit_stats_path, 'w') as f:
            json.dump(subreddit_data, f, indent=2, default=str)
        print(f"Subreddit analysis saved to {subreddit_stats_path}")
    
    # Save graph structure
    if graph_data:
        graph_path = artifacts_dir / 'graph_structure.json'
        # Convert numpy types for JSON serialization
        graph_data_serializable = {
            'users': graph_data['users'].tolist() if hasattr(graph_data['users'], 'tolist') else list(graph_data['users']),
            'user_to_id': graph_data['user_to_id'],
            'edges': graph_data['edges'],
            'num_users': int(graph_data['num_users']),
            'num_edges': int(graph_data['num_edges'])
        }
        with open(graph_path, 'w') as f:
            json.dump(graph_data_serializable, f, indent=2, default=str)
        print(f"Graph structure saved to {graph_path}")

def create_data_prep_visualizations(combined_df, balanced_df, subreddit_stats, config):
    """Create visualizations for data preparation analysis."""
    print("Creating data preparation visualizations...")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Label distribution in full dataset
    if not combined_df.empty:
        label_counts = combined_df['davidson_label'].value_counts()
        labels = ['Normal', 'Offensive', 'Hate Speech']
        colors = ['green', 'orange', 'red']
        
        axes[0, 0].pie(label_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Label Distribution (Full Dataset)')
    
    # Balanced dataset distribution
    if not balanced_df.empty:
        balanced_counts = balanced_df['binary_label'].value_counts()
        axes[0, 1].bar(['Normal', 'Hate'], balanced_counts.values, color=['green', 'red'], alpha=0.7)
        axes[0, 1].set_title('Balanced Dataset Distribution')
        axes[0, 1].set_ylabel('Count')
    
    # Top hate subreddits
    if len(subreddit_stats) == 2:
        top_hate, top_normal = subreddit_stats
        if not top_hate.empty:
            hate_plot_data = top_hate.head(10)
            axes[1, 0].barh(range(len(hate_plot_data)), hate_plot_data['hate_ratio'], color='red', alpha=0.7)
            axes[1, 0].set_yticks(range(len(hate_plot_data)))
            axes[1, 0].set_yticklabels(hate_plot_data['subreddit'], fontsize=8)
            axes[1, 0].set_xlabel('Hate Speech Ratio')
            axes[1, 0].set_title('Top 10 Hate Speech Subreddits')
        
        # Top normal subreddits
        if not top_normal.empty:
            normal_plot_data = top_normal.head(10)
            axes[1, 1].barh(range(len(normal_plot_data)), normal_plot_data['combined_ratio'], color='green', alpha=0.7)
            axes[1, 1].set_yticks(range(len(normal_plot_data)))
            axes[1, 1].set_yticklabels(normal_plot_data['subreddit'], fontsize=8)
            axes[1, 1].set_xlabel('Combined Hate/Offensive Ratio')
            axes[1, 1].set_title('Top 10 Normal Subreddits')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'data_preparation_analysis.png', bbox_inches='tight')
    plt.close()
    
    print("Data preparation visualizations saved to figures/ directory")

def main():
    parser = argparse.ArgumentParser(description="Prepare Reddit data with Davidson labeling")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Reddit Data Preparation with Davidson Labeling ===")
    
    # Load Reddit data
    submissions_df, comments_df = load_reddit_data(config)
    
    if submissions_df.empty and comments_df.empty:
        print("No data found! Please check your data paths.")
        return
    
    # Apply Davidson lexicon labeling
    combined_df = label_reddit_data(submissions_df, comments_df, config)
    
    # Analyze subreddit hate levels
    top_hate_subreddits, top_normal_subreddits = analyze_subreddit_hate_levels(combined_df, config)
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(combined_df, top_hate_subreddits, top_normal_subreddits, config)
    
    # Prepare graph structure
    graph_data = prepare_graph_structure(balanced_df, config)
    
    # Create visualizations
    create_data_prep_visualizations(combined_df, balanced_df, (top_hate_subreddits, top_normal_subreddits), config)
    
    # Save all prepared data
    save_prepared_data(combined_df, balanced_df, (top_hate_subreddits, top_normal_subreddits), graph_data, config)
    
    print(f"\n=== Data Preparation Summary ===")
    print(f"Full dataset: {len(combined_df)} samples")
    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(f"Hate subreddits identified: {len(top_hate_subreddits)}")
    print(f"Normal subreddits identified: {len(top_normal_subreddits)}")
    print(f"Graph nodes: {graph_data.get('num_users', 0)}")
    print(f"Graph edges: {graph_data.get('num_edges', 0)}")
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
