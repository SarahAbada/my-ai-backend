from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from lstm_model import train_lstm

app = FastAPI()

# Load CSV
data = pd.read_csv("data/reviews.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Naive Bayes ---
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
preds_nb = nb.predict(X_test_vec)
acc1 = accuracy_score(y_test, preds_nb)
print(f"Naive Bayes Accuracy: {acc1:.4f}")
print(classification_report(y_test, preds_nb))
print(confusion_matrix(y_test, preds_nb))

# --- Linear SVM ---
svm = LinearSVC(max_iter=10000)
svm.fit(X_train_vec, y_train)
preds_svm = svm.predict(X_test_vec)
acc2 = accuracy_score(y_test, preds_svm)
print(f"Linear SVM Accuracy: {acc2:.4f}")

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)
preds_lr = lr.predict(X_test_vec)
acc3 = accuracy_score(y_test, preds_lr)
print(f"Logistic Regression Accuracy: {acc3:.4f}")

# --- PyTorch LSTM ---
lstm_model, acc4, lstm_vectorizer = train_lstm(data["text"], data["label"])
print(f"LSTM Accuracy: {acc4:.4f}")

# --- Performance comparison chart ---
plt.bar(["Naive Bayes", "SVM", "LogReg", "LSTM"], [acc1, acc2, acc3, acc4])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()