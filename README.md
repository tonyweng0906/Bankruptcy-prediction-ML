# Corporate Bankruptcy Prediction Analysis

## Project Overview
This project uses machine learning techniques to predict corporate bankruptcy risk. By analyzing a dataset containing 95 financial indicators, we built a logistic regression model to identify key financial features that may lead to corporate bankruptcy. The project implements a complete workflow including data preprocessing, feature selection, model training, and evaluation.

## Dataset
- **Source**: Corporate bankruptcy data (Bankrupt data.csv)
- **Size**: 6,819 records × 96 features
- **Target Variable**: `Bankrupt?` (binary: 0 = not bankrupt, 1 = bankrupt)
- **Features**: 95 financial indicators covering:
  - Profitability metrics
  - Return on assets
  - Gross margins
  - Debt ratios
  - Cash flow indicators
  - Turnover ratios, etc.

## Project Structure

### 1. Data Preparation
- Data loading and cleaning
- Data splitting: Training set (3000), Test set (2000), Validation set (1819)
- Feature standardization

### 2. Class Imbalance Handling
- RandomOverSampler for random oversampling
- SMOTE algorithm to enhance minority class samples
- Final data sizes:
  - Training set: 4,366 records
  - Test set: 2,905 records
  - Validation set: 2,626 records

### 3. Logistic Regression Models
- Base logistic regression model
- L1-regularized logistic regression model
- Cross-validation for optimal regularization parameter (C=0.017)
- Class weight balancing

### 4. Feature Selection
Automatic feature selection via L1 regularization identified 22 key financial indicators including:
- ROA(C) before interest and depreciation before interest
- Continuous interest rate (after tax)
- Net Value Per Share (B)
- Persistent EPS in the Last Four Seasons
- And 18 other significant financial metrics

### 5. Model Evaluation
Model performance across three datasets:

#### Training Set:
- Accuracy: 89.78%
- F1-Score (Bankrupt class): 85.89%

#### Validation Set:
- Accuracy: 85.38%
- F1-Score (Bankrupt class): 79.29%

#### Test Set:
- Accuracy: 85.47%
- F1-Score (Bankrupt class): 78.67%

## Key Technologies
1. **Data Preprocessing**: Standardization, oversampling, SMOTE
2. **Feature Selection**: Automatic selection via L1 regularization
3. **Model Optimization**: Regularization parameter tuning
4. **Evaluation Metrics**: Accuracy, precision, recall, F1-score

## Installation & Dependencies

```bash
# Install required packages
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
