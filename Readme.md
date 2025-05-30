# Email Spam Classification using Machine Learning

This project implements email spam classification using two different machine learning approaches: Logistic Regression and K-means Clustering. The project includes data processing, model training, performance evaluation, and result visualization.

##  Features

- Email spam data analysis and processing
- Logistic Regression model training
- K-means clustering implementation
- Model performance evaluation with multiple metrics
- Result visualization through various plots
- Model persistence for future use

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Adjusted Rand Index (for clustering)
- Confusion Matrix

##  Visualizations

The project generates the following visualizations:
- Confusion matrices for both models
- Accuracy comparison between models
- Actual label distribution in clusters
- K-means clustering in 2D space (after PCA dimensionality reduction)

##  Technologies Used

- Python 3.x
- Pandas: Data processing
- Scikit-learn: Machine Learning
- Matplotlib & Seaborn: Visualization
- Joblib: Model persistence

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/thanhluu0312/CLASSIFYING-SPAM-EMAILS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

##  Usage

1. Prepare the data:
   - Place the `spambase.csv` file in the project root directory

2. Run the program:
```bash
python "CLASSIFYING SPAM EMAILS.py"
```

3. Results:
   - Evaluation metrics will be printed to the console
   - Visualization plots will be saved as PNG files
   - Trained models will be saved as .pkl files

##  Project Structure

- `CLASSIFYING SPAM EMAILS.py`: Main source code file
- `spambase.csv`: Original dataset
- `logistic_regression_model.pkl`: Trained logistic regression model
- `kmeans_model.pkl`: Trained K-means clustering model
- `scaler.pkl`: Data scaler
- Visualization output files:
  - `logistic_confusion_matrix.png`
  - `kmeans_confusion_matrix.png`
  - `accuracy_comparison.png`
  - `actual_labels_distribution.png`
  - `kmeans_pca_2d.png`


