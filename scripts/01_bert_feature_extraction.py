#!/usr/bin/env python3
"""
BERT Feature Extraction Script for Reddit Hate Speech Analysis.
Extracts BERT embeddings for posts and comments to create rich text representations.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class BERTFeatureExtractor:
    """BERT feature extractor for Reddit text data."""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=512, batch_size=32):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading BERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"BERT model loaded successfully!")
    
    def extract_embeddings(self, texts):
        """Extract BERT embeddings for a list of texts."""
        embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting BERT embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def extract_user_features(self, df):
        """Extract user-level features from Reddit data."""
        print("Extracting user features...")
        
        user_features = {}
        
        for user, group in tqdm(df.groupby('author'), desc="Processing users"):
            if pd.isna(user):
                continue
                
            # Basic activity features
            total_posts = len(group)
            hate_posts = len(group[group.get('davidson_label', 0) == 2])
            offensive_posts = len(group[group.get('davidson_label', 0) == 1])
            
            # Temporal features
            if 'created_utc' in group.columns:
                timestamps = pd.to_datetime(group['created_utc'], unit='s')
                time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # hours
                avg_posting_interval = time_span / total_posts if total_posts > 1 else 0
            else:
                time_span = 0
                avg_posting_interval = 0
            
            # Subreddit diversity
            unique_subreddits = group['subreddit'].nunique()
            
            # Text features
            avg_text_length = group['text_content'].str.len().mean() if 'text_content' in group.columns else 0
            
            user_features[user] = {
                'total_posts': total_posts,
                'hate_posts': hate_posts,
                'offensive_posts': offensive_posts,
                'hate_ratio': hate_posts / total_posts if total_posts > 0 else 0,
                'offensive_ratio': offensive_posts / total_posts if total_posts > 0 else 0,
                'time_span_hours': time_span,
                'avg_posting_interval_hours': avg_posting_interval,
                'subreddit_diversity': unique_subreddits,
                'avg_text_length': avg_text_length
            }
        
        return user_features

def load_prepared_data(config):
    """Load prepared data from data preparation step."""
    print("Loading prepared data...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    
    # Load balanced dataset
    balanced_path = artifacts_dir / 'balanced_dataset.parquet'
    full_path = artifacts_dir / 'full_labeled_data.parquet'
    
    if balanced_path.exists():
        balanced_df = pd.read_parquet(balanced_path)
        print(f"Loaded balanced dataset: {len(balanced_df)} samples")
    else:
        balanced_df = pd.DataFrame()
        print("No balanced dataset found")
    
    if full_path.exists():
        full_df = pd.read_parquet(full_path)
        print(f"Loaded full dataset: {len(full_df)} samples")
    else:
        full_df = pd.DataFrame()
        print("No full dataset found")
    
    return balanced_df, full_df

def create_text_embeddings(df, extractor, config):
    """Create BERT embeddings for text data."""
    print("Creating BERT embeddings for text content...")
    
    if df.empty or 'text_content' not in df.columns:
        print("No text content found for embedding extraction")
        return df, np.array([])
    
    # Clean and prepare texts
    texts = df['text_content'].fillna('').astype(str).tolist()
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(texts)
    
    # Add embedding info to dataframe
    df = df.copy()
    df['has_embedding'] = True
    df['embedding_dim'] = embeddings.shape[1] if len(embeddings) > 0 else 0
    
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1] if len(embeddings) > 0 else 0}")
    
    return df, embeddings

def save_bert_features(df, embeddings, user_features, config):
    """Save BERT features and embeddings."""
    print("Saving BERT features...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save enhanced dataframe
    df.to_parquet(artifacts_dir / 'bert_enhanced_dataset.parquet')
    print(f"Saved enhanced dataset: {len(df)} samples")
    
    # Save embeddings as numpy array
    if len(embeddings) > 0:
        np.save(artifacts_dir / 'bert_embeddings.npy', embeddings)
        print(f"Saved embeddings: {embeddings.shape}")
    
    # Save user features
    with open(artifacts_dir / 'user_features.json', 'w') as f:
        json.dump(user_features, f, indent=2, default=str)
    print(f"Saved user features for {len(user_features)} users")
    
    # Save feature summary
    feature_summary = {
        'embedding_model': config.get('bert', {}).get('model_name', 'distilbert-base-uncased'),
        'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
        'num_samples': len(df),
        'num_users': len(user_features),
        'has_embeddings': len(embeddings) > 0
    }
    
    with open(artifacts_dir / 'bert_feature_summary.json', 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    print("BERT feature extraction completed!")

def main():
    parser = argparse.ArgumentParser(description="BERT feature extraction for Reddit data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== BERT Feature Extraction ===")
    
    # Load prepared data
    balanced_df, full_df = load_prepared_data(config)
    
    # Use balanced dataset for feature extraction (more manageable)
    df = balanced_df if not balanced_df.empty else full_df
    
    if df.empty:
        print("No data found! Please run data preparation first.")
        return
    
    # Initialize BERT extractor
    bert_config = config.get('bert', {})
    extractor = BERTFeatureExtractor(
        model_name=bert_config.get('model_name', 'distilbert-base-uncased'),
        max_length=bert_config.get('max_length', 512),
        batch_size=bert_config.get('batch_size', 32)
    )
    
    # Extract user features
    user_features = extractor.extract_user_features(df)
    
    # Create text embeddings
    df_enhanced, embeddings = create_text_embeddings(df, extractor, config)
    
    # Save results
    save_bert_features(df_enhanced, embeddings, user_features, config)
    
    print(f"\n=== BERT Feature Extraction Summary ===")
    print(f"Processed samples: {len(df_enhanced)}")
    print(f"Generated embeddings: {len(embeddings)}")
    print(f"User features extracted: {len(user_features)}")
    print(f"Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 0}")

if __name__ == "__main__":
    main()
