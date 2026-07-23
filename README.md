# Fake News Detection Using Machine Learning

## Project Overview

This project uses supervised machine learning to detect fake news and explore patterns related to geography, news topics, political events, and article characteristics.

The analysis was based on a Kaggle dataset containing 4,000 news articles and 24 variables. （https://www.kaggle.com/datasets/khushikyad001/fake-news-detection)

## Research Questions

- Which U.S. states are more associated with fake news?
- Are certain news topics more likely to be fake?
- Does fake news increase during major political events?
- Which machine learning model performs best for classification?

## Methodology

We cleaned and selected relevant features, encoded categorical variables, and split the data into training, validation, and test sets.

The following models were compared:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- KNN

After hyperparameter tuning, the best-performing model was an **SVM with an RBF kernel and `C = 0.1`**.

## Key Findings

- `sentiment_score`, `word_count`, and `num_shares` were the most influential features.
- Entertainment had the highest fake-news ratio at 54%, followed by Sports at 51% and Business at 50%.
- The fake-news ratio during the U.S. presidential election period was 42%, compared with 51% outside the election period.
- The final model achieved relatively high recall but lower precision, indicating that misclassification remained a challenge.

## Limitations and Future Work

The model did not use article titles or full text, which limited its ability to capture language patterns.

Future improvements could include:

- TF-IDF
- BERT embeddings
- Random Forest or Gradient Boosting
- A larger and more representative dataset


## Technologies Used

- **Programming Language:** Python
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Data Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook

## Project Files

- [Code Works](code.ipynb)
- [Fake_News_Dataset.csv](fake_news_dataset.csv)

