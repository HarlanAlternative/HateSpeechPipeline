#!/usr/bin/env python3
"""
Graph construction script for Reddit data analysis.
Builds user interaction graphs from submissions and comments.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import yaml
from collections import defaultdict, Counter

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load submissions and comments data."""
    print("Loading data...")
    
    mini_dir = Path(config['paths']['mini_dir'])
    
    # Load submissions
    sub_parts_file = mini_dir / "submissions_parts.txt"
    if sub_parts_file.exists():
        sub_parts = sub_parts_file.read_text().strip().split('\n')
        sub_parts = [p for p in sub_parts if p.strip()]
        
        max_files = min(config['read']['max_submissions_files'], len(sub_parts))
        sub_data = []
        
        for part_file in sub_parts[:max_files]:
            part_path = mini_dir / part_file
            if part_path.exists():
                df = pd.read_parquet(part_path)
                sub_data.append(df)
        
        if sub_data:
            submissions_df = pd.concat(sub_data, ignore_index=True)
            print(f"Loaded {len(submissions_df)} submissions")
        else:
            submissions_df = pd.DataFrame()
    else:
        submissions_df = pd.DataFrame()
    
    # Load comments
    com_parts_file = mini_dir / "comments_parts.txt"
    if com_parts_file.exists():
        com_parts = com_parts_file.read_text().strip().split('\n')
        com_parts = [p for p in com_parts if p.strip()]
        
        max_files = min(config['read']['max_comments_files'], len(com_parts))
        com_data = []
        
        for part_file in com_parts[:max_files]:
            part_path = mini_dir / part_file
            if part_path.exists():
                df = pd.read_parquet(part_path)
                com_data.append(df)
        
        if com_data:
            comments_df = pd.concat(com_data, ignore_index=True)
            print(f"Loaded {len(comments_df)} comments")
        else:
            comments_df = pd.DataFrame()
    else:
        comments_df = pd.DataFrame()
    
    return submissions_df, comments_df

def filter_users(df, min_degree=2):
    """Filter out deleted users and bots."""
    if df.empty:
        return df
    
    # Filter out deleted users and common bots
    filtered_df = df[
        (~df['author'].isin(['[deleted]', '[removed]', 'AutoModerator'])) &
        (df['author'].notna()) &
        (df['author'] != '')
    ].copy()
    
    return filtered_df

def build_reply_graph(submissions_df, comments_df, config):
    """Build reply-based interaction graph with optimized performance."""
    print("Building reply graph...")
    
    G = nx.DiGraph()
    
    # Filter users
    submissions_df = filter_users(submissions_df)
    comments_df = filter_users(comments_df)
    
    if submissions_df.empty and comments_df.empty:
        return G
    
    # Create submission author mapping
    sub_authors = {}
    for _, row in submissions_df.iterrows():
        sub_authors[row['id']] = row['author']
        sub_authors[f"t3_{row['id']}"] = row['author']  # Reddit format
    
    # Filter valid comments first
    valid_comments = comments_df[
        (comments_df['author'].notna()) & 
        (~comments_df['author'].isin(['[deleted]', '[removed]']))
    ].copy()
    
    print(f"Processing {len(valid_comments)} valid comments...")
    
    # Create comment author mapping for faster lookup
    comment_authors = {}
    for _, comment in valid_comments.iterrows():
        comment_authors[comment['id']] = comment['author']
    
    # Add reply edges with optimized lookup
    reply_count = 0
    chunk_size = 5000  # Smaller chunks for better progress tracking
    
    for i in range(0, len(valid_comments), chunk_size):
        chunk = valid_comments.iloc[i:i+chunk_size]
        
        for _, comment in chunk.iterrows():
            author = comment['author']
            parent_id = comment['parent_id']
            
            # Find parent author with optimized lookup
            parent_author = None
            
            if parent_id.startswith('t3_'):
                # Parent is a submission
                sub_id = parent_id[3:]  # Remove 't3_' prefix
                parent_author = sub_authors.get(sub_id)
            elif parent_id.startswith('t1_'):
                # Parent is a comment - use pre-built mapping
                parent_comment_id = parent_id[3:]
                parent_author = comment_authors.get(parent_comment_id)
            
            if parent_author and parent_author != author:
                G.add_edge(parent_author, author)
                reply_count += 1
        
        # Progress reporting
        progress = (i + chunk_size) / len(valid_comments) * 100
        print(f"Progress: {progress:.1f}% - Processed {min(i + chunk_size, len(valid_comments))} comments, {reply_count} edges added")
    
    print(f"Added {reply_count} reply edges")
    return G

def build_cocomment_graph(submissions_df, comments_df, config):
    """Build co-comment graph (users commenting on same submissions) with size limits."""
    print("Building co-comment graph...")
    
    G = nx.Graph()
    
    # Filter users
    comments_df = filter_users(comments_df)
    
    if comments_df.empty:
        return G
    
    # Group comments by submission
    submission_comments = defaultdict(list)
    for _, comment in comments_df.iterrows():
        link_id = comment['link_id']
        author = comment['author']
        if pd.notna(author) and author not in ['[deleted]', '[removed]']:
            submission_comments[link_id].append(author)
    
    # Add co-comment edges with size limits to prevent explosion
    cocomment_count = 0
    processed_threads = 0
    max_threads = 1000  # Limit number of threads to process
    max_authors_per_thread = 50  # Limit authors per thread
    
    for link_id, authors in submission_comments.items():
        if processed_threads >= max_threads:
            break
            
        # Skip very large threads
        unique_authors = list(set(authors))
        if len(unique_authors) > max_authors_per_thread:
            continue
        
        # Create edges between all pairs of authors
        for i in range(len(unique_authors)):
            for j in range(i + 1, len(unique_authors)):
                G.add_edge(unique_authors[i], unique_authors[j])
                cocomment_count += 1
        
        processed_threads += 1
        
        if processed_threads % 100 == 0:
            print(f"Processed {processed_threads} threads, {cocomment_count} edges added")
    
    print(f"Added {cocomment_count} co-comment edges from {processed_threads} threads")
    return G

def analyze_graph(G, config):
    """Analyze graph properties."""
    print("Analyzing graph...")
    
    if G.number_of_nodes() == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'is_connected': False,
            'num_components': 0,
            'largest_component_size': 0,
            'avg_degree': 0.0,
            'density': 0.0
        }
    
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Connectivity
    if G.is_directed():
        is_connected = nx.is_weakly_connected(G)
        components = list(nx.weakly_connected_components(G))
    else:
        is_connected = nx.is_connected(G)
        components = list(nx.connected_components(G))
    
    largest_component_size = max(len(comp) for comp in components) if components else 0
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0
    
    # Density
    density = nx.density(G)
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'is_connected': is_connected,
        'num_components': len(components),
        'largest_component_size': largest_component_size,
        'avg_degree': float(avg_degree),
        'density': float(density)
    }
    
    return stats

def create_degree_plot(G, config):
    """Create degree distribution plot."""
    print("Creating degree distribution plot...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    if G.number_of_nodes() == 0:
        print("No nodes in graph, skipping degree plot")
        return
    
    degrees = [d for n, d in G.degree()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'degree_hist.png', bbox_inches='tight')
    plt.close()

def filter_largest_component(G, max_nodes):
    """Filter to largest connected component and top degree nodes."""
    if G.number_of_nodes() == 0:
        return G
    
    print(f"Filtering graph to max {max_nodes} nodes...")
    
    # Get largest component
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    if not components:
        return G
    
    largest_component = max(components, key=len)
    G_largest = G.subgraph(largest_component).copy()
    
    # If still too large, take top degree nodes
    if G_largest.number_of_nodes() > max_nodes:
        degrees = [(n, d) for n, d in G_largest.degree()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, d in degrees[:max_nodes]]
        G_filtered = G_largest.subgraph(top_nodes).copy()
    else:
        G_filtered = G_largest
    
    print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    return G_filtered

def main():
    parser = argparse.ArgumentParser(description="Build interaction graph")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Graph Construction ===")
    
    # Load data
    submissions_df, comments_df = load_data(config)
    
    if submissions_df.empty and comments_df.empty:
        print("No data available for graph construction!")
        return
    
    # Build graphs
    reply_G = build_reply_graph(submissions_df, comments_df, config)
    cocomment_G = build_cocomment_graph(submissions_df, comments_df, config)
    
    # Combine graphs (use reply graph as primary)
    G = reply_G.copy()
    G.add_edges_from(cocomment_G.edges())
    
    print(f"Combined graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter to manageable size
    G_filtered = filter_largest_component(G, config['graph']['max_nodes'])
    
    # Analyze graph
    stats = analyze_graph(G_filtered, config)
    
    # Create visualizations
    create_degree_plot(G_filtered, config)
    
    # Save graph and statistics
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save graph
    graph_path = artifacts_dir / 'graph_user.gpickle'
    import pickle
    with open(graph_path, 'wb') as f:
        pickle.dump(G_filtered, f)
    
    # Save statistics
    stats_path = artifacts_dir / 'graph_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n=== Graph Statistics ===")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Connected: {stats['is_connected']}")
    print(f"Components: {stats['num_components']}")
    print(f"Largest component: {stats['largest_component_size']} nodes")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Density: {stats['density']:.4f}")
    
    print(f"\nGraph saved to {graph_path}")
    print(f"Statistics saved to {stats_path}")
    print("Graph construction completed successfully!")

if __name__ == "__main__":
    main()
