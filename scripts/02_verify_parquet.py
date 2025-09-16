#!/usr/bin/env python3
"""
Parquet verification script for Reddit data analysis.
Verifies sliced data and generates summary statistics.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def verify_submissions(config):
    """Verify submissions data and generate statistics."""
    print("Verifying submissions...")
    
    mini_dir = Path(config['paths']['mini_dir'])
    parts_file = mini_dir / "submissions_parts.txt"
    
    if not parts_file.exists():
        print("No submissions parts found!")
        return None
    
    parts = parts_file.read_text().strip().split('\n')
    parts = [p for p in parts if p.strip()]
    
    print(f"Found {len(parts)} submission parts")
    
    # Read first few parts for verification
    max_files = min(config['read']['max_submissions_files'], len(parts))
    all_data = []
    
    for i, part_file in enumerate(parts[:max_files]):
        part_path = mini_dir / part_file
        if part_path.exists():
            df = pd.read_parquet(part_path)
            all_data.append(df)
            print(f"Read {len(df)} rows from {part_file}")
    
    if not all_data:
        print("No submission data found!")
        return None
    
    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total submissions: {len(combined_df)}")
    
    # Generate statistics
    stats = {
        'total_rows': len(combined_df),
        'columns': list(combined_df.columns),
        'min_created_utc': int(combined_df['created_utc'].min()),
        'max_created_utc': int(combined_df['created_utc'].max()),
        'subreddit_counts': combined_df['subreddit'].value_counts().head(20).to_dict(),
        'avg_score': float(combined_df['score'].mean()),
        'total_parts': len(parts)
    }
    
    return combined_df, stats

def verify_comments(config):
    """Verify comments data and generate statistics."""
    print("Verifying comments...")
    
    mini_dir = Path(config['paths']['mini_dir'])
    parts_file = mini_dir / "comments_parts.txt"
    
    if not parts_file.exists():
        print("No comments parts found!")
        return None
    
    parts = parts_file.read_text().strip().split('\n')
    parts = [p for p in parts if p.strip()]
    
    print(f"Found {len(parts)} comment parts")
    
    # Read first few parts for verification
    max_files = min(config['read']['max_comments_files'], len(parts))
    all_data = []
    
    for i, part_file in enumerate(parts[:max_files]):
        part_path = mini_dir / part_file
        if part_path.exists():
            df = pd.read_parquet(part_path)
            all_data.append(df)
            print(f"Read {len(df)} rows from {part_file}")
    
    if not all_data:
        print("No comment data found!")
        return None
    
    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total comments: {len(combined_df)}")
    
    # Generate statistics
    stats = {
        'total_rows': len(combined_df),
        'columns': list(combined_df.columns),
        'min_created_utc': int(combined_df['created_utc'].min()),
        'max_created_utc': int(combined_df['created_utc'].max()),
        'subreddit_counts': combined_df['subreddit'].value_counts().head(20).to_dict(),
        'avg_score': float(combined_df['score'].mean()),
        'total_parts': len(parts)
    }
    
    return combined_df, stats

def create_plots(sub_df, com_df, config):
    """Create visualization plots."""
    print("Creating plots...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    # Subreddit distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if sub_df is not None:
        sub_counts = sub_df['subreddit'].value_counts().head(10)
        sub_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Top 10 Subreddits (Submissions)')
        ax1.set_xlabel('Subreddit')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    if com_df is not None:
        com_counts = com_df['subreddit'].value_counts().head(10)
        com_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Top 10 Subreddits (Comments)')
        ax2.set_xlabel('Subreddit')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'subreddit_bar.png', bbox_inches='tight')
    plt.close()
    
    # Timestamp histograms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if sub_df is not None:
        sub_df['created_utc'].hist(bins=50, ax=ax1, alpha=0.7, color='skyblue')
        ax1.set_title('Submission Timestamps Distribution')
        ax1.set_xlabel('Created UTC')
        ax1.set_ylabel('Count')
    
    if com_df is not None:
        com_df['created_utc'].hist(bins=50, ax=ax2, alpha=0.7, color='lightcoral')
        ax2.set_title('Comment Timestamps Distribution')
        ax2.set_xlabel('Created UTC')
        ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'timestamp_hist.png', bbox_inches='tight')
    plt.close()
    
    print("Plots saved to figures/ directory")

def main():
    parser = argparse.ArgumentParser(description="Verify parquet data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Parquet Verification ===")
    
    # Verify submissions
    sub_result = verify_submissions(config)
    if sub_result:
        sub_df, sub_stats = sub_result
    else:
        sub_df, sub_stats = None, None
    
    # Verify comments
    com_result = verify_comments(config)
    if com_result:
        com_df, com_stats = com_result
    else:
        com_df, com_stats = None, None
    
    # Create plots
    create_plots(sub_df, com_df, config)
    
    # Save summary
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    summary = {
        'submissions': sub_stats,
        'comments': com_stats,
        'verification_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(artifacts_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    if sub_stats:
        print(f"Submissions: {sub_stats['total_rows']} rows, {sub_stats['total_parts']} parts")
        print(f"  Time range: {sub_stats['min_created_utc']} - {sub_stats['max_created_utc']}")
        print(f"  Top subreddit: {list(sub_stats['subreddit_counts'].keys())[0]}")
    
    if com_stats:
        print(f"Comments: {com_stats['total_rows']} rows, {com_stats['total_parts']} parts")
        print(f"  Time range: {com_stats['min_created_utc']} - {com_stats['max_created_utc']}")
        print(f"  Top subreddit: {list(com_stats['subreddit_counts'].keys())[0]}")
    
    print("Verification completed successfully!")

if __name__ == "__main__":
    main()
