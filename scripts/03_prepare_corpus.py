#!/usr/bin/env python3
"""
Corpus preparation script for Reddit data analysis.
Creates weakly labeled text corpus from submissions and comments.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def clean_text(text, config):
    """Clean text according to configuration."""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    if config['corpus']['lowercase']:
        text = text.lower()
    
    if config['corpus']['strip_urls']:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    if config['corpus']['strip_markdown']:
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
    
    if config['corpus']['strip_code']:
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_weak_labels(text):
    """Create weak labels based on simple keyword matching."""
    # Simple keyword-based labeling for demonstration
    positive_keywords = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'like', 'enjoy', 'happy', 'positive', 'best', 'awesome',
        'support', 'agree', 'yes', 'true', 'correct', 'right'
    ]
    
    negative_keywords = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
        'angry', 'sad', 'negative', 'worst', 'stupid', 'wrong',
        'against', 'disagree', 'no', 'false', 'incorrect', 'problem'
    ]
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_keywords if word in text_lower)
    neg_count = sum(1 for word in negative_keywords if word in text_lower)
    
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return 0
    else:
        return 0  # Default to negative for neutral

def prepare_submissions_corpus(config):
    """Prepare corpus from submissions."""
    print("Preparing submissions corpus...")
    
    mini_dir = Path(config['paths']['mini_dir'])
    parts_file = mini_dir / "submissions_parts.txt"
    
    if not parts_file.exists():
        print("No submissions parts found!")
        return pd.DataFrame()
    
    parts = parts_file.read_text().strip().split('\n')
    parts = [p for p in parts if p.strip()]
    
    max_files = min(config['read']['max_submissions_files'], len(parts))
    all_data = []
    
    for part_file in parts[:max_files]:
        part_path = mini_dir / part_file
        if part_path.exists():
            df = pd.read_parquet(part_path)
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Prepare text corpus
    corpus_data = []
    text_fields = config['corpus']['text_fields_submission']
    
    for _, row in combined_df.iterrows():
        # Combine text fields
        text_parts = []
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                text_parts.append(str(row[field]))
        
        text = ' '.join(text_parts)
        text = clean_text(text, config)
        
        if len(text) >= config['corpus']['min_doc_len']:
            label = create_weak_labels(text)
            corpus_data.append({
                'text': text,
                'label': label,
                'created_utc': row['created_utc'],
                'subreddit': row['subreddit'],
                'source': 'submission',
                'original_id': row['id']
            })
    
    return pd.DataFrame(corpus_data)

def prepare_comments_corpus(config):
    """Prepare corpus from comments."""
    print("Preparing comments corpus...")
    
    mini_dir = Path(config['paths']['mini_dir'])
    parts_file = mini_dir / "comments_parts.txt"
    
    if not parts_file.exists():
        print("No comments parts found!")
        return pd.DataFrame()
    
    parts = parts_file.read_text().strip().split('\n')
    parts = [p for p in parts if p.strip()]
    
    max_files = min(config['read']['max_comments_files'], len(parts))
    all_data = []
    
    for part_file in parts[:max_files]:
        part_path = mini_dir / part_file
        if part_path.exists():
            df = pd.read_parquet(part_path)
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sample comments if needed
    if config['slice']['comment_sample_frac'] < 1.0:
        sample_size = int(len(combined_df) * config['slice']['comment_sample_frac'])
        combined_df = combined_df.sample(n=sample_size, random_state=config['random_state'])
    
    # Prepare text corpus
    corpus_data = []
    text_field = config['corpus']['text_field_comment']
    
    for _, row in combined_df.iterrows():
        text = str(row[text_field]) if pd.notna(row[text_field]) else ''
        text = clean_text(text, config)
        
        if len(text) >= config['corpus']['min_doc_len']:
            label = create_weak_labels(text)
            corpus_data.append({
                'text': text,
                'label': label,
                'created_utc': row['created_utc'],
                'subreddit': row['subreddit'],
                'source': 'comment',
                'original_id': row['id']
            })
    
    return pd.DataFrame(corpus_data)

def create_label_balance_plot(corpus_df, config):
    """Create label balance visualization."""
    print("Creating label balance plot...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall label distribution
    label_counts = corpus_df['label'].value_counts()
    label_counts.plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
    ax1.set_title('Overall Label Distribution')
    ax1.set_xlabel('Label (0=Negative, 1=Positive)')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Negative', 'Positive'], rotation=0)
    
    # Label distribution by source
    source_label = corpus_df.groupby(['source', 'label']).size().unstack(fill_value=0)
    source_label.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
    ax2.set_title('Label Distribution by Source')
    ax2.set_xlabel('Source')
    ax2.set_ylabel('Count')
    ax2.legend(['Negative', 'Positive'])
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'label_balance.png', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Prepare text corpus")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Corpus Preparation ===")
    
    # Prepare submissions corpus
    sub_corpus = prepare_submissions_corpus(config)
    print(f"Submissions corpus: {len(sub_corpus)} documents")
    
    # Prepare comments corpus
    com_corpus = prepare_comments_corpus(config)
    print(f"Comments corpus: {len(com_corpus)} documents")
    
    # Combine corpora
    if not sub_corpus.empty and not com_corpus.empty:
        corpus_df = pd.concat([sub_corpus, com_corpus], ignore_index=True)
    elif not sub_corpus.empty:
        corpus_df = sub_corpus
    elif not com_corpus.empty:
        corpus_df = com_corpus
    else:
        print("No corpus data created!")
        return
    
    print(f"Total corpus: {len(corpus_df)} documents")
    
    # Sort by timestamp
    corpus_df = corpus_df.sort_values('created_utc').reset_index(drop=True)
    
    # Create visualizations
    create_label_balance_plot(corpus_df, config)
    
    # Save corpus
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    corpus_path = artifacts_dir / 'corpus.parquet'
    corpus_df.to_parquet(corpus_path, engine='pyarrow')
    
    print(f"Corpus saved to {corpus_path}")
    
    # Print summary
    print(f"\n=== Corpus Summary ===")
    print(f"Total documents: {len(corpus_df)}")
    print(f"Positive labels: {sum(corpus_df['label'] == 1)} ({sum(corpus_df['label'] == 1)/len(corpus_df)*100:.1f}%)")
    print(f"Negative labels: {sum(corpus_df['label'] == 0)} ({sum(corpus_df['label'] == 0)/len(corpus_df)*100:.1f}%)")
    print(f"Submissions: {sum(corpus_df['source'] == 'submission')}")
    print(f"Comments: {sum(corpus_df['source'] == 'comment')}")
    print(f"Average text length: {corpus_df['text'].str.len().mean():.1f} characters")
    
    print("Corpus preparation completed successfully!")

if __name__ == "__main__":
    main()
