# Reddit Data Analysis Pipeline

A comprehensive data analysis pipeline for Reddit data, featuring data preprocessing, text classification, network analysis, and visualization.

## Project Overview

This project provides a complete end-to-end pipeline for analyzing Reddit data, from raw JSON files to publication-ready results. It includes sentiment analysis, user interaction modeling, and information diffusion visualization.

### Key Features
- **Scalable Data Processing**: Handles large Reddit datasets efficiently
- **Machine Learning Pipeline**: TF-IDF + Logistic Regression baseline model
- **Network Analysis**: User interaction graphs and community detection
- **Visualization**: Comprehensive charts and plots for analysis
- **One-Click Execution**: Automated batch scripts for easy deployment

## Data Source

- **Data Source**: Academic Torrents Reddit 2024-12
- **Time Range**: December 1-7, 2024 (one week of data)
- **Target Subreddits**: politics, worldnews, AskReddit
- **File Format**: JSON Lines (.json)

## Environment Requirements

### Recommended Environment
- **Python**: 3.10+ (recommended to use conda environment)
- **Memory**: 16GB+ RAM
- **Storage**: SSD recommended

### Dependency Installation

```bash
# Basic dependencies
pip install -U pip wheel setuptools

# Core dependencies
pip install "pyarrow>=17" "fastparquet>=2024.5" pandas tqdm scikit-learn matplotlib networkx joblib pydantic ruamel.yaml seaborn
```

### Conda Environment (Recommended)

```bash
conda create -n at310 python=3.10 -y
conda activate at310
pip install "pyarrow>=17" "fastparquet>=2024.5" pandas tqdm scikit-learn matplotlib networkx joblib pydantic ruamel.yaml seaborn
```

## Project Structure

```
project_root/
├── configs/                 # Configuration files
│   ├── exp_small.yaml      # Small sample configuration
│   └── exp_full.yaml       # Full dataset configuration
├── scripts/                 # Analysis scripts
│   ├── 00_env_check.py     # Environment check
│   ├── 01_slice_dataset.py # Data slicing
│   ├── 02_verify_parquet.py# Data verification
│   ├── 03_prepare_corpus.py# Corpus preparation
│   ├── 04_baseline_tfidf_lr.py # Baseline model
│   ├── 05_build_graph.py   # Graph construction
│   ├── 06_visualize_diffusion.py # Diffusion visualization
│   └── 07_eval_report.py   # Evaluation report
├── artifacts/              # Intermediate outputs
├── figures/                # Chart outputs
├── mini_dataset/           # Sliced data
├── run_small.bat          # Small sample one-click run
├── run_full.bat           # Full dataset one-click run
└── README.md              # This document
```

## Quick Start

### Small Sample Mode (Recommended for first run)

```bash
# Windows
run_small.bat

# Or run manually
python scripts/00_env_check.py --config configs/exp_small.yaml
python scripts/01_slice_dataset.py --config configs/exp_small.yaml
# ... other scripts
```

**Expected Time**: 30-60 minutes  
**Data Volume**: 10K submissions, 200K comments

### Full Dataset Mode

```bash
# Windows
run_full.bat

# Or run manually
python scripts/00_env_check.py --config configs/exp_full.yaml
python scripts/01_slice_dataset.py --config configs/exp_full.yaml
# ... other scripts
```

**Expected Time**: Several hours  
**Data Volume**: Full week dataset

## Analysis Pipeline

### 1. Environment Check (`00_env_check.py`)
- Check Python version and required packages
- Automatically install missing packages
- Verify parquet engine availability

### 2. Data Slicing (`01_slice_dataset.py`)
- Filter raw JSON data by subreddit and time
- Write sliced data to Parquet format
- Generate sliced files for submissions and comments

### 3. Data Verification (`02_verify_parquet.py`)
- Verify sliced data integrity
- Generate data statistics summary
- Create data distribution charts

### 4. Corpus Preparation (`03_prepare_corpus.py`)
- Build weakly-labeled text corpus
- Text cleaning and preprocessing
- Proxy label generation based on keywords

### 5. Baseline Model (`04_baseline_tfidf_lr.py`)
- TF-IDF feature extraction
- Logistic regression classifier
- Time-series data splitting
- Performance evaluation and visualization

### 6. Graph Construction (`05_build_graph.py`)
- User interaction graph construction
- Reply relationships and co-comment relationships
- Graph statistics analysis and degree distribution

### 7. Diffusion Visualization (`06_visualize_diffusion.py`)
- Hot discussion thread analysis
- Hierarchical trees and force-directed graphs
- Discussion depth and time span analysis

### 8. Evaluation Report (`07_eval_report.py`)
- Summarize all analysis results
- Generate PPT bullet points
- Create data tables

## Output Files

### artifacts/ Directory
- `summary.json` - Data statistics summary
- `corpus.parquet` - Text corpus
- `baseline_metrics.json` - Baseline model metrics
- `baseline_model.joblib` - Trained model
- `graph_user.gpickle` - User interaction graph
- `graph_stats.json` - Graph statistics
- `thread_*_stats.json` - Hot thread statistics
- `table_numbers.csv` - Summary data table
- `slide_bullets.md` - PPT bullet points

### figures/ Directory
- `subreddit_bar.png` - Subreddit distribution
- `timestamp_hist.png` - Timestamp distribution
- `label_balance.png` - Label balance
- `confusion_matrix.png` - Confusion matrix
- `pr_curve.png` - Precision-recall curve
- `feature_top_terms.png` - Feature importance
- `degree_hist.png` - Degree distribution
- `thread_tree_*.png` - Discussion trees
- `thread_force_*.png` - Force-directed graphs

## Configuration

### Small Sample Configuration (exp_small.yaml)
- Limited data volume for quick testing
- Suitable for development and debugging
- Lower memory requirements

### Full Dataset Configuration (exp_full.yaml)
- Process complete week dataset
- Suitable for production analysis
- Requires more computational resources

## Important Notes

### Weak Labeling
The current text labels are **proxy labels/weak labels** generated based on simple keyword matching, used only for demonstration purposes. For formal analysis, it is recommended to:
- Use human-annotated real data
- Use public annotation datasets
- Implement more complex labeling methods

### Data Privacy
- Only processes publicly available Reddit data
- Follows Academic Torrents usage terms
- Does not store personal sensitive information

### Performance Optimization
- Uses Parquet format for improved I/O efficiency
- Chunked processing to avoid memory overflow
- Configurable data volume limits

## Troubleshooting

### Common Issues

1. **Insufficient Memory**
   - Use small sample mode
   - Reduce chunksize parameter
   - Increase virtual memory

2. **Parquet Engine Errors**
   - Run environment check script
   - Reinstall pyarrow/fastparquet
   - Check Python version compatibility

3. **Missing Data Files**
   - Confirm original JSON file paths are correct
   - Check file name format
   - Verify file permissions

### Logging and Debugging
- All scripts have detailed console output
- Error messages show specific failure steps
- Can run each script individually for debugging

## Extension and Customization

### Adding New Analysis Modules
1. Create new script in `scripts/` directory
2. Update configuration files to add new parameters
3. Modify batch files to include new steps

### Modifying Data Sources
1. Update file paths in configuration files
2. Adjust data filtering conditions
3. Modify field mapping logic

### Custom Visualization
1. Modify matplotlib style configuration
2. Add new chart types
3. Adjust chart size and DPI

## Results Summary

### Data Processing Results
- **Processed**: 4,869 Reddit submissions
- **Extracted**: 200,000 related comments
- **Corpus**: 185,776 documents with weak labels

### Model Performance
- **Accuracy**: 92.0%
- **F1-Score**: 86.0% (macro), 92.5% (weighted)
- **AUC**: 96.6%

### Network Analysis
- **Users**: 20,000 nodes (filtered from 64,688)
- **Interactions**: 266,467 edges
- **Components**: 29 connected components
- **Average Degree**: 26.65

## License

This project is based on Academic Torrents Reddit data and follows the corresponding usage terms.

## Contact

For questions or suggestions, please submit an Issue through the project repository.