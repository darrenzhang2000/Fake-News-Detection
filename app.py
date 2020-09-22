import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data
# Assumes that news.csv is in the same file as this one
df=pd.read_csv('./news.csv')

# Get shape and head
# print(df.shape)
# print(df.head())

# Get the labels
labels=df.label
# print(labels.head())


# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# print(tfidf_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
# print(f'Accuracy: {round(score*100,2)}%')

title = "Darren's Amazing Super Powers"
text = "I have 300 IQ, I am 50 feet tall and can shoot lasers out of my eyes. I am the greatest of them all and everyone refer to me as Darren The Great."

# print(pac.predict([[text]]))

f = open('input.txt', "r")
input_data = f.read()
vectorized_input_data = tfidf_vectorizer.transform([input_data])
prediction = pac.predict(vectorized_input_data)
print(prediction)