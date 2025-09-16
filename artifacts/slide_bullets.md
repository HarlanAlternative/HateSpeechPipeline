# Reddit Data Analysis Pipeline - Initial Results

## Methodology & Data Processing

### Data Collection & Preprocessing
• Processed 4,869 Reddit submissions from December 2024
• Extracted 200,000 related comments
• Filtered by target subreddits: politics, worldnews, AskReddit
• Time window: 2024-12-01 to 2024-12-07 (one week)

### Text Corpus Construction
• Created weakly-labeled corpus: 185,776 documents
• Label distribution: 27,701 positive, 158,075 negative
• Text preprocessing: lowercase, URL removal, markdown cleaning
• Average document length: 191 characters
• **Note: Labels are proxy/weak labels for demonstration purposes**

## Baseline Results

### Text Classification Performance
• **Accuracy**: 0.920
• **F1-Score (Macro)**: 0.860
• **F1-Score (Weighted)**: 0.925
• **AUC**: 0.966
• Model: TF-IDF + Logistic Regression
• Features: 50K max features, 1-2 grams
• Split: 70% train, 15% validation, 15% test (time-based)

## Network Analysis

### User Interaction Graph
• **Nodes**: 1,993 users
• **Edges**: 11,024 interactions
• **Connected Components**: 558
• **Average Degree**: 11.06
• **Graph Density**: 0.0056
• Edge types: co-comment interactions
• Filtered: removed deleted users, bots, low-degree nodes

## Key Findings & Next Steps

### Initial Observations
• Successfully processed Reddit data with scalable pipeline
• Weak labeling approach provides reasonable baseline performance
• User interaction patterns show clear community structure
• Discussion threads exhibit hierarchical organization

### Limitations & Future Work
• **Weak Labels**: Current labels are proxy-based, need human annotation
• **Temporal Analysis**: Limited to one week, need longer timeframes
• **Content Analysis**: Basic text features, need advanced NLP
• **Diffusion Modeling**: Need formal information diffusion models
• **Scalability**: Current pipeline handles small samples, need optimization

### Proposed Next Steps
• Implement human annotation interface for ground truth labels
• Extend to full month dataset for temporal analysis
• Add advanced text features (embeddings, sentiment, topics)
• Develop information diffusion prediction models
• Scale pipeline for real-time processing

---
*Analysis based on Reddit data from Academic Torrents (2024-12)*
*Pipeline: Data slicing → Text corpus → Baseline ML → Network analysis → Visualization*