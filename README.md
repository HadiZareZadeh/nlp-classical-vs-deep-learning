# Classical Machine Learning vs Deep Learning for Text Classification

## Project Description

This project compares classical machine learning approaches (TF-IDF + Logistic Regression) with deep learning (Neural Networks) for binary sentiment classification on the IMDb movie reviews dataset. The goal is to understand when each approach is preferable and the trade-offs involved.

**Why This Project Matters**:
- **Practical Decision Making**: Helps practitioners choose between classical ML and deep learning based on their specific needs
- **Resource Awareness**: Demonstrates computational and data requirements for different approaches
- **Interpretability Trade-offs**: Shows when interpretability (classical ML) vs. performance (deep learning) matters
- **Real-world Application**: Sentiment analysis is widely used in industry, making this comparison highly relevant
- **Baseline Understanding**: Establishes when simple models are sufficient vs. when complexity is needed

**Key Research Questions**:
- When should you use classical ML vs. deep learning for text classification?
- What are the computational trade-offs between approaches?
- How do preprocessing strategies differ between classical ML and deep learning?
- What level of performance improvement does deep learning provide, and is it worth the added complexity?

## Dataset Description

**Dataset Name**: IMDb Movie Reviews Dataset

**Source**: Stanford AI Lab / TensorFlow Datasets / HuggingFace Datasets

**Dataset Details**:
- **Number of samples**: 50,000 movie reviews
  - Training set: 25,000 reviews
  - Test set: 25,000 reviews
- **Features**: Movie review text (variable length, typically 100-2000 words per review)
- **Target variable**: Sentiment label (binary classification)
  - Positive sentiment: 1
  - Negative sentiment: 0
- **Task**: Binary sentiment classification (predicting positive or negative sentiment)
- **Class distribution**: Balanced (50% positive, 50% negative)
- **Vocabulary**: Top 10,000 most frequent words (when using TensorFlow version)

**Why This Dataset**:
- **Well-established benchmark**: Standard dataset for sentiment analysis research, widely used in NLP literature
- **Binary classification**: Clear positive/negative sentiment labels make it ideal for comparing approaches
- **Text complexity**: Reviews contain varied vocabulary, length, and writing style, representing real-world text data
- **Real-world relevance**: Sentiment analysis is a practical NLP application used in industry
- **Size**: Large enough (~50,000 reviews) to demonstrate scalability differences between classical ML and deep learning
- **Balanced classes**: Equal distribution ensures fair evaluation without class imbalance issues

**Data Loading**:
```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000, skip_top=0, maxlen=None
)
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real IMDb Movie Reviews dataset loaded from TensorFlow's `keras.datasets.imdb.load_data()` function.

## Project Structure

```
project2_nlp_classical_vs_deep/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_classical_ml_tfidf.ipynb
    ├── 03_deep_learning_nn.ipynb
    └── 04_comparison_analysis.ipynb
```

## Key Implementations

### Classical ML Approach
- Text preprocessing pipeline (tokenization, cleaning, normalization)
- TF-IDF vectorization
- Logistic Regression classifier
- Feature analysis and interpretation

### Deep Learning Approach
- Text preprocessing for neural networks
- Word embeddings (or tokenization)
- Feedforward Neural Network using TensorFlow/Keras
- Training with early stopping and validation

## Comparison Metrics

- **Accuracy**: Overall classification performance
- **Training Time**: Computational efficiency
- **Overfitting Behavior**: Generalization capability
- **Interpretability**: Model explainability
- **Resource Requirements**: Memory and computational needs

## Learning Objectives

1. **Text Preprocessing**: Understand different preprocessing strategies for ML vs DL
2. **Feature Engineering**: Compare hand-crafted features (TF-IDF) vs learned representations
3. **Model Complexity**: Trade-offs between simple and complex models
4. **When to Use What**: Guidelines for choosing classical ML vs deep learning

## Results

The notebooks demonstrate:
- Classical ML often performs well with less data and computation
- Deep learning can capture complex patterns but requires more resources
- TF-IDF provides interpretable features
- Neural networks learn distributed representations

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter notebooks in order (01 → 04)
3. Run all cells to reproduce experiments
4. Note: First run will download the IMDb dataset

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Scikit-learn
- NLTK or spaCy
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter

