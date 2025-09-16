#!/usr/bin/env python3
"""
Dataset slicing script for Reddit data analysis.
Filters and chunks Reddit submissions and comments data.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import re

def parse_iso8601_to_epoch(iso_str):
    """Convert ISO8601 string to epoch timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except:
        return None

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def filter_submissions(config):
    """Filter and slice submissions data."""
    print("Processing submissions...")
    
    raw_path = Path(config['paths']['raw_submissions'])
    mini_dir = Path(config['paths']['mini_dir'])
    mini_dir.mkdir(exist_ok=True)
    
    # Parse time filters
    start_epoch = parse_iso8601_to_epoch(config['filters']['start_utc'])
    end_epoch = parse_iso8601_to_epoch(config['filters']['end_utc'])
    target_subreddits = set(config['filters']['subreddits'])
    
    print(f"Time range: {start_epoch} - {end_epoch}")
    print(f"Target subreddits: {target_subreddits}")
    
    # Process submissions
    submission_parts = []
    total_rows = 0
    part_num = 0
    max_submissions = config['slice']['max_submissions']
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading submissions")):
            if max_submissions and total_rows >= max_submissions:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Filter by subreddit
                if data.get('subreddit', '').lower() not in target_subreddits:
                    continue
                
                # Filter by time
                created_utc = data.get('created_utc')
                if not created_utc or not (start_epoch <= created_utc <= end_epoch):
                    continue
                
                # Extract required fields
                row = {
                    'id': data.get('id', ''),
                    'author': data.get('author', ''),
                    'title': data.get('title', ''),
                    'selftext': data.get('selftext', ''),
                    'subreddit': data.get('subreddit', ''),
                    'created_utc': created_utc,
                    'score': data.get('score', 0)
                }
                
                submission_parts.append(row)
                total_rows += 1
                
                # Write chunk when full
                if len(submission_parts) >= config['slice']['sub_shard_rows']:
                    part_path = mini_dir / f"submissions_part{part_num:03d}.parquet"
                    pd.DataFrame(submission_parts).to_parquet(part_path, engine='pyarrow')
                    print(f"Wrote {len(submission_parts)} submissions to {part_path}")
                    submission_parts = []
                    part_num += 1
                    
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    # Write remaining data
    if submission_parts:
        part_path = mini_dir / f"submissions_part{part_num:03d}.parquet"
        pd.DataFrame(submission_parts).to_parquet(part_path, engine='pyarrow')
        print(f"Wrote {len(submission_parts)} submissions to {part_path}")
        part_num += 1
    
    # Save parts list
    parts_list = [f"submissions_part{i:03d}.parquet" for i in range(part_num)]
    with open(mini_dir / "submissions_parts.txt", 'w') as f:
        f.write('\n'.join(parts_list))
    
    print(f"Total submissions processed: {total_rows}")
    print(f"Created {part_num} submission parts")
    
    return total_rows, part_num

def filter_comments(config, submission_ids):
    """Filter and slice comments data based on submission IDs."""
    print("Processing comments...")
    
    raw_path = Path(config['paths']['raw_comments'])
    mini_dir = Path(config['paths']['mini_dir'])
    
    # Parse time filters
    start_epoch = parse_iso8601_to_epoch(config['filters']['start_utc'])
    end_epoch = parse_iso8601_to_epoch(config['filters']['end_utc'])
    target_subreddits = set(config['filters']['subreddits'])
    
    # Create ID sets for fast lookup
    id_set = set(submission_ids)
    t3_id_set = set(f"t3_{sid}" for sid in submission_ids)
    all_ids = id_set | t3_id_set
    
    print(f"Looking for comments with link_id in {len(all_ids)} submission IDs")
    
    # Process comments
    comment_parts = []
    total_rows = 0
    part_num = 0
    max_comments = config['slice']['max_comments']
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading comments")):
            if max_comments and total_rows >= max_comments:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Filter by subreddit
                if data.get('subreddit', '').lower() not in target_subreddits:
                    continue
                
                # Filter by time
                created_utc = data.get('created_utc')
                if not created_utc or not (start_epoch <= created_utc <= end_epoch):
                    continue
                
                # Filter by link_id
                link_id = data.get('link_id', '')
                if link_id not in all_ids:
                    continue
                
                # Extract required fields
                row = {
                    'id': data.get('id', ''),
                    'link_id': link_id,
                    'parent_id': data.get('parent_id', ''),
                    'author': data.get('author', ''),
                    'body': data.get('body', ''),
                    'subreddit': data.get('subreddit', ''),
                    'created_utc': created_utc,
                    'score': data.get('score', 0)
                }
                
                comment_parts.append(row)
                total_rows += 1
                
                # Write chunk when full
                if len(comment_parts) >= config['slice']['com_shard_rows']:
                    part_path = mini_dir / f"comments_part{part_num:03d}.parquet"
                    pd.DataFrame(comment_parts).to_parquet(part_path, engine='pyarrow')
                    print(f"Wrote {len(comment_parts)} comments to {part_path}")
                    comment_parts = []
                    part_num += 1
                    
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    # Write remaining data
    if comment_parts:
        part_path = mini_dir / f"comments_part{part_num:03d}.parquet"
        pd.DataFrame(comment_parts).to_parquet(part_path, engine='pyarrow')
        print(f"Wrote {len(comment_parts)} comments to {part_path}")
        part_num += 1
    
    # Save parts list
    parts_list = [f"comments_part{i:03d}.parquet" for i in range(part_num)]
    with open(mini_dir / "comments_parts.txt", 'w') as f:
        f.write('\n'.join(parts_list))
    
    print(f"Total comments processed: {total_rows}")
    print(f"Created {part_num} comment parts")
    
    return total_rows, part_num

def main():
    parser = argparse.ArgumentParser(description="Slice Reddit dataset")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Dataset Slicing ===")
    
    # Process submissions first
    sub_count, sub_parts = filter_submissions(config)
    
    # Read submission IDs for comment filtering
    mini_dir = Path(config['paths']['mini_dir'])
    submission_ids = []
    
    for part_file in (mini_dir / "submissions_parts.txt").read_text().strip().split('\n'):
        if part_file:
            part_path = mini_dir / part_file
            df = pd.read_parquet(part_path)
            submission_ids.extend(df['id'].tolist())
    
    print(f"Collected {len(submission_ids)} submission IDs")
    
    # Process comments
    com_count, com_parts = filter_comments(config, submission_ids)
    
    print(f"\n=== Summary ===")
    print(f"Submissions: {sub_count} rows, {sub_parts} parts")
    print(f"Comments: {com_count} rows, {com_parts} parts")
    print("Dataset slicing completed successfully!")

if __name__ == "__main__":
    main()
