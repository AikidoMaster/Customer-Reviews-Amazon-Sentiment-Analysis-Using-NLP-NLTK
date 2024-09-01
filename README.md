

# Customer Review Analysis Using NLP/NLTK

## Overview

This project focuses on sentiment analysis of customer reviews using Natural Language Processing (NLP) techniques, specifically leveraging the Natural Language Toolkit (NLTK) in Python. Sentiment analysis, also known as opinion mining, is a key component in understanding public opinion, consumer preferences, and market trends. This project demonstrates the process of building sentiment analysis models from data preprocessing to model evaluation.

![Project Overview](https://s6.ezgif.com/tmp/ezgif-6-d0f6254513.gif)



Dataset link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download


## Techniques and Approaches

The notebook includes the following approaches and techniques:

1. **Data Preprocessing**:
   - **Tokenization**: Splitting text into individual words (tokens).
   - **Stop Words Removal**: Filtering out common words that do not contribute to sentiment (e.g., "is", "the").
   - **Lemmatization**: Reducing words to their base or root form.

2. **Sentiment Analysis Approaches**:
   - **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon-based sentiment analysis tool specifically attuned to sentiments expressed in social media.
   - **Roberta Pretrained Model**: A transformer model from Hugging Face used for sentiment classification.
   - **Hugging Face Pipeline**: Utilized for easy implementation of sentiment analysis with pre-trained models.

3. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of review scores.
   - Understanding the text data through tokenization and frequency distribution.

4. **Sentiment Classification**:
   - Building and training models to classify the sentiment of customer reviews.
   - Predicting the sentiment of new, unseen text data.

5. **Model Evaluation**:
   - Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Comparing the effectiveness of different models and approaches.

## Project Structure

- **Step 0: Read in Data and NLTK Basics**
  - Load the dataset and perform initial data exploration.
  - Basic operations with NLTK, including tokenization and stop word removal.
  
- **Step 1: Exploratory Data Analysis (EDA)**
  - Visualize the distribution of review scores.
  - Perform tokenization and understand the structure of the text data.

- **Step 2: Data Preprocessing**
  - Implement lemmatization to prepare text for sentiment analysis.
  - Use NLTK functions to clean and process the text data.

- **Step 3: Sentiment Analysis Using VADER**
  - Apply the VADER model to the dataset to obtain sentiment scores.

- **Step 4: Sentiment Analysis Using Pretrained Models**
  - Leverage Hugging Face's `Roberta` model and pipeline for advanced sentiment classification.

- **Step 5: Model Evaluation**
  - Evaluate the performance of each model and approach using standard metrics.

## Dependencies

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- NLTK
- Hugging Face's `transformers`

Ensure you have all the required libraries installed:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers
```

## How to Run the Notebook

1. Clone the repository and navigate to the project directory.
2. Install the dependencies using the provided `requirements.txt` file.
3. Open the notebook in Jupyter Notebook or any compatible environment.
4. Run each cell sequentially to reproduce the results.

## Conclusion

This project provides a comprehensive guide to performing sentiment analysis on customer reviews using NLP techniques. By following along, you will gain practical experience with text processing, model building, and evaluation, making this an essential resource for anyone interested in sentiment analysis and NLP.

## Acknowledgments

This project utilizes various tools and libraries such as NLTK and Hugging Face's Transformers, which have been instrumental in developing robust NLP models.

---
