#!/usr/bin/env python3
"""
Diffusion visualization script for Reddit data analysis.
Visualizes discussion threads and information diffusion patterns.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import yaml
from collections import defaultdict
import matplotlib.patches as mpatches

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

def find_hot_threads(submissions_df, comments_df, top_n=2):
    """Find threads with most comments."""
    print(f"Finding top {top_n} hot threads...")
    
    if comments_df.empty:
        return []
    
    # Count comments per submission
    comment_counts = comments_df['link_id'].value_counts()
    top_threads = comment_counts.head(top_n)
    
    print(f"Top threads by comment count:")
    for link_id, count in top_threads.items():
        print(f"  {link_id}: {count} comments")
    
    return top_threads.index.tolist()

def build_thread_tree(submission_id, submissions_df, comments_df):
    """Build discussion tree for a specific submission."""
    print(f"Building discussion tree for {submission_id}...")
    
    # Get submission info
    submission = submissions_df[submissions_df['id'] == submission_id]
    if submission.empty:
        # Try with t3_ prefix
        submission = submissions_df[submissions_df['id'] == submission_id[3:]]
    
    if submission.empty:
        print(f"Submission {submission_id} not found!")
        return None, None
    
    submission = submission.iloc[0]
    
    # Get all comments for this submission
    thread_comments = comments_df[comments_df['link_id'] == submission_id].copy()
    
    if thread_comments.empty:
        print(f"No comments found for {submission_id}")
        return None, None
    
    # Build tree structure
    G = nx.DiGraph()
    
    # Add submission as root
    G.add_node('root', 
               author=submission['author'],
               text=submission.get('title', '')[:100] + '...',
               created_utc=submission['created_utc'],
               score=submission['score'],
               node_type='submission')
    
    # Add comments
    for _, comment in thread_comments.iterrows():
        if pd.isna(comment['author']) or comment['author'] in ['[deleted]', '[removed]']:
            continue
        
        G.add_node(comment['id'],
                   author=comment['author'],
                   text=comment['body'][:100] + '...' if len(comment['body']) > 100 else comment['body'],
                   created_utc=comment['created_utc'],
                   score=comment['score'],
                   node_type='comment')
        
        # Add edge to parent
        parent_id = comment['parent_id']
        if parent_id.startswith('t3_'):
            # Parent is submission
            G.add_edge('root', comment['id'])
        elif parent_id.startswith('t1_'):
            # Parent is comment
            parent_comment_id = parent_id[3:]
            if parent_comment_id in G.nodes:
                G.add_edge(parent_comment_id, comment['id'])
    
    return G, submission

def create_thread_tree_plot(G, submission, thread_id, config):
    """Create hierarchical tree visualization."""
    print("Creating thread tree plot...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    if G.number_of_nodes() == 0:
        print("Empty graph, skipping tree plot")
        return
    
    # Use hierarchical layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    submission_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'submission']
    comment_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'comment']
    
    # Draw submission node
    if submission_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=submission_nodes, 
                              node_color='red', node_size=500, alpha=0.8)
    
    # Draw comment nodes
    if comment_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=comment_nodes, 
                              node_color='lightblue', node_size=200, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=10)
    
    # Add labels for important nodes
    important_nodes = ['root'] + list(G.nodes)[:10]  # Root + first 10 nodes
    labels = {n: G.nodes[n]['author'][:10] for n in important_nodes if n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f'Discussion Tree: {thread_id}\n'
              f'Title: {submission.get("title", "")[:50]}...\n'
              f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
    plt.axis('off')
    
    # Add legend
    submission_patch = mpatches.Patch(color='red', label='Submission')
    comment_patch = mpatches.Patch(color='lightblue', label='Comments')
    plt.legend(handles=[submission_patch, comment_patch])
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'thread_tree_{thread_id.replace("t3_", "")}.png', 
                bbox_inches='tight', dpi=150)
    plt.close()

def create_force_directed_plot(G, submission, thread_id, config):
    """Create force-directed layout visualization."""
    print("Creating force-directed plot...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    
    if G.number_of_nodes() == 0:
        print("Empty graph, skipping force-directed plot")
        return
    
    # Use force-directed layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=100)
    except:
        pos = nx.random_layout(G)
    
    plt.figure(figsize=(12, 10))
    
    # Color nodes by timestamp (early = blue, late = red)
    timestamps = [G.nodes[n]['created_utc'] for n in G.nodes]
    min_time, max_time = min(timestamps), max(timestamps)
    
    node_colors = []
    for n in G.nodes:
        if G.nodes[n]['node_type'] == 'submission':
            node_colors.append('red')
        else:
            # Color by relative time
            rel_time = (G.nodes[n]['created_utc'] - min_time) / (max_time - min_time)
            node_colors.append(plt.cm.viridis(rel_time))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=8)
    
    plt.title(f'Force-Directed Layout: {thread_id}\n'
              f'Title: {submission.get("title", "")[:50]}...\n'
              f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
    plt.axis('off')
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min_time, vmax=max_time))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestamp')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'thread_force_{thread_id.replace("t3_", "")}.png', 
                bbox_inches='tight', dpi=150)
    plt.close()

def analyze_thread_stats(G, submission, thread_id):
    """Analyze thread statistics."""
    if G.number_of_nodes() == 0:
        return {
            'thread_id': thread_id,
            'num_comments': 0,
            'num_users': 0,
            'max_depth': 0,
            'time_span_hours': 0,
            'avg_score': 0.0
        }
    
    # Count comments and users
    comment_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'comment']
    users = set(G.nodes[n]['author'] for n in G.nodes)
    
    # Calculate depth
    try:
        depths = nx.single_source_shortest_path_length(G, 'root')
        max_depth = max(depths.values()) if depths else 0
    except:
        max_depth = 0
    
    # Time span
    timestamps = [G.nodes[n]['created_utc'] for n in G.nodes]
    time_span_hours = (max(timestamps) - min(timestamps)) / 3600 if timestamps else 0
    
    # Average score
    scores = [G.nodes[n]['score'] for n in G.nodes if 'score' in G.nodes[n]]
    avg_score = np.mean(scores) if scores else 0.0
    
    stats = {
        'thread_id': thread_id,
        'num_comments': len(comment_nodes),
        'num_users': len(users),
        'max_depth': max_depth,
        'time_span_hours': float(time_span_hours),
        'avg_score': float(avg_score)
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Visualize diffusion patterns")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Diffusion Visualization ===")
    
    # Load data
    submissions_df, comments_df = load_data(config)
    
    if submissions_df.empty and comments_df.empty:
        print("No data available for visualization!")
        return
    
    # Find hot threads
    hot_threads = find_hot_threads(submissions_df, comments_df, top_n=2)
    
    if not hot_threads:
        print("No hot threads found!")
        return
    
    # Analyze each hot thread
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    all_stats = []
    
    for thread_id in hot_threads:
        print(f"\nProcessing thread: {thread_id}")
        
        # Build discussion tree
        G, submission = build_thread_tree(thread_id, submissions_df, comments_df)
        
        if G is None:
            continue
        
        # Create visualizations
        create_thread_tree_plot(G, submission, thread_id, config)
        create_force_directed_plot(G, submission, thread_id, config)
        
        # Analyze statistics
        stats = analyze_thread_stats(G, submission, thread_id)
        all_stats.append(stats)
        
        print(f"  Comments: {stats['num_comments']}")
        print(f"  Users: {stats['num_users']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Time span: {stats['time_span_hours']:.1f} hours")
        print(f"  Avg score: {stats['avg_score']:.1f}")
    
    # Save statistics
    for stats in all_stats:
        thread_id = stats['thread_id'].replace('t3_', '')
        stats_path = artifacts_dir / f'thread_{thread_id}_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(all_stats)} hot threads")
    print(f"Visualizations saved to figures/ directory")
    print(f"Statistics saved to artifacts/ directory")
    print("Diffusion visualization completed successfully!")

if __name__ == "__main__":
    main()
