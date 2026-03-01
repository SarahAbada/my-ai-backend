from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

app = FastAPI()

# Load CSV
data = pd.read_csv("data/reviews.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2)

# Vectorize and train
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

plt.bar(["Naive Bayes", "SVM", "LogReg"], [acc1, acc2, acc3])
plt.show()