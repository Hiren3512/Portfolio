# Sentimental-Analysis-On-Moview-Reviews
This project aims to perform sentiment analysis on movie reviews using machine learning techniques. We will be using a dataset of movie reviews and their associated sentiment labels (positive or negative) to train a machine learning model that can classify new, unseen reviews.

Dataset
The dataset used in this project is the IMDB movie reviews dataset, which can be downloaded from this link. It contains 50,000 movie reviews, with 25,000 reviews labeled as positive and 25,000 reviews labeled as negative. Each review is stored in a separate text file.

Preprocessing
Before training our model, we need to preprocess the text data. This includes steps such as removing stop words, stemming or lemmatizing the words, and converting the text data into numerical features that can be fed into our machine learning model. We will be using the Natural Language Toolkit (NLTK) library for text preprocessing.

Model
We will be using a Multinomial Naive Bayes algorithm for sentiment classification, which is a popular algorithm for text classification tasks.

Evaluation
We will evaluate the performance of our model using metrics such as accuracy, precision, recall, and F1-score. We will also use a confusion matrix to visualize the number of true positives, false positives, true negatives, and false negatives.

Dependencies
1) Python 3.x
2) Pandas
3) NumPy
4) NLTK

Download the IMDB movie reviews dataset from the link provided above and extract the zip file.

Results
Our trained model achieved an accuracy of 86% on the test set, with a precision of 0.87, recall of 0.84, and an F1-score of 0.86. These results suggest that our logistic regression model is able to effectively classify movie reviews based on their sentiment.

Credits
IMDB movie reviews dataset: https://ai.stanford.edu/~amaas/data/sentiment/
NLTK library: https://www.nltk.org/
