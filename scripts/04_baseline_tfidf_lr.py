#!/usr/bin/env python3
"""
Baseline text classification script using TF-IDF and Logistic Regression.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score
)
import joblib
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_corpus(config):
    """Load prepared corpus."""
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    corpus_path = artifacts_dir / 'corpus.parquet'
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    
    return pd.read_parquet(corpus_path)

def split_data(corpus_df, config):
    """Split data into train/val/test sets."""
    print("Splitting data...")
    
    # Sort by timestamp for time-based split
    corpus_df = corpus_df.sort_values('created_utc').reset_index(drop=True)
    
    n_total = len(corpus_df)
    n_train = int(n_total * config['split']['train_ratio'])
    n_val = int(n_total * config['split']['val_ratio'])
    
    train_df = corpus_df[:n_train]
    val_df = corpus_df[n_train:n_train + n_val]
    test_df = corpus_df[n_train + n_val:]
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def train_baseline_model(train_df, val_df, config):
    """Train TF-IDF + Logistic Regression baseline."""
    print("Training baseline model...")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=config['baseline']['max_features'],
        ngram_range=tuple(config['baseline']['ngram_range']),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print(f"TF-IDF features: {X_train.shape[1]}")
    
    # Logistic Regression
    clf = LogisticRegression(
        C=config['baseline']['C'],
        class_weight=config['baseline']['class_weight'],
        random_state=config['random_state'],
        max_iter=1000
    )
    
    clf.fit(X_train, train_df['label'])
    
    # Validation predictions
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    
    return vectorizer, clf, X_train, X_val, y_val_pred, y_val_proba

def evaluate_model(clf, vectorizer, test_df, config):
    """Evaluate model on test set."""
    print("Evaluating model...")
    
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']
    
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, y_test_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    metrics = {
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score'],
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'auc': float(auc_score),
        'accuracy': report['accuracy']
    }
    
    return metrics, y_test_pred, y_test_proba, report

def create_visualizations(y_test, y_test_pred, y_test_proba, config):
    """Create evaluation visualizations."""
    print("Creating visualizations...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use(config['plots']['style'])
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    
    # Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Negative', 'Positive'])
    ax1.set_yticklabels(['Negative', 'Positive'])
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    ax2.plot(recall, precision, 'b-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # PR Curve (separate plot)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'pr_curve.png', bbox_inches='tight')
    plt.close()

def create_feature_analysis(clf, vectorizer, config):
    """Create feature importance analysis."""
    print("Creating feature analysis...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = clf.coef_[0]
    
    # Get top positive and negative features
    top_positive_idx = np.argsort(coefficients)[-20:][::-1]
    top_negative_idx = np.argsort(coefficients)[:20]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top positive features
    pos_features = [feature_names[i] for i in top_positive_idx]
    pos_coeffs = [coefficients[i] for i in top_positive_idx]
    
    ax1.barh(range(len(pos_features)), pos_coeffs, color='lightblue')
    ax1.set_yticks(range(len(pos_features)))
    ax1.set_yticklabels(pos_features)
    ax1.set_xlabel('Coefficient')
    ax1.set_title('Top 20 Positive Features')
    ax1.invert_yaxis()
    
    # Top negative features
    neg_features = [feature_names[i] for i in top_negative_idx]
    neg_coeffs = [coefficients[i] for i in top_negative_idx]
    
    ax2.barh(range(len(neg_features)), neg_coeffs, color='lightcoral')
    ax2.set_yticks(range(len(neg_features)))
    ax2.set_yticklabels(neg_features)
    ax2.set_xlabel('Coefficient')
    ax2.set_title('Top 20 Negative Features')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'feature_top_terms.png', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Baseline Model Training ===")
    
    # Load corpus
    corpus_df = load_corpus(config)
    print(f"Loaded corpus: {len(corpus_df)} documents")
    
    # Split data
    train_df, val_df, test_df = split_data(corpus_df, config)
    
    # Train model
    vectorizer, clf, X_train, X_val, y_val_pred, y_val_proba = train_baseline_model(train_df, val_df, config)
    
    # Evaluate on test set
    metrics, y_test_pred, y_test_proba, report = evaluate_model(clf, vectorizer, test_df, config)
    
    # Create visualizations
    create_visualizations(test_df['label'], y_test_pred, y_test_proba, config)
    create_feature_analysis(clf, vectorizer, config)
    
    # Save model and metrics
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / 'baseline_model.joblib'
    joblib.dump({'vectorizer': vectorizer, 'classifier': clf}, model_path)
    
    # Save metrics
    metrics_path = artifacts_dir / 'baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")
    print(f"Precision (macro): {metrics['precision_macro']:.3f}")
    print(f"Recall (macro): {metrics['recall_macro']:.3f}")
    
    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print("Baseline training completed successfully!")

if __name__ == "__main__":
    main()
