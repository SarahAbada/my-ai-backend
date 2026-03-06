# Sentiment Analyzer API

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

## 🚀 Overview

**Sentiment Analyzer** is a high-performance AI backend designed to classify textual reviews as **positive** or **negative** in real-time. Built with **FastAPI** for speed and **Scikit-Learn** for robust machine learning, this project demonstrates a complete end-to-end ML pipeline—from raw data processing to model deployment.

This project solves the problem of understanding user feedback at scale, automating the analysis of thousands of reviews to derive actionable insights instantly.

## ✨ Key Features

-   **⚡ High-Speed API**: RESTful endpoints powered by FastAPI for sub-millisecond inference latency.
-   **🧠 Multi-Model Machine Learning**: Implements and benchmarks multiple algorithms including:
    -   Multinomial Naive Bayes (current default)
    -   Linear Support Vector Machines (SVM)
    -   Logistic Regression
    -   **PyTorch LSTM** (sequential deep-learning classifier)
-   **📊 Data Visualization**: Built-in performance comparison tools using Matplotlib to visualize model accuracy and confusion matrices.
-   **🛠️ Robust Preprocessing**: Efficient text vectorization using `CountVectorizer` to handle large vocabularies (IMDb dataset).

## 🛠️ Tech Stack

-   **Language**: Python
-   **API Framework**: FastAPI, Uvicorn
-   **Machine Learning**: Scikit-Learn, Pandas, NumPy
-   **Deep Learning**: PyTorch (LSTM classifier)
-   **Visualization**: Matplotlib
-   **Data Source**: IMDb Large Movie Review Dataset (50k+ reviews)

## 🏎️ Getting Started

### Prerequisites

-   Python 3.9+
-   pip

### Requirements

The following key libraries are required (full list in `requirements.txt`):

| Library | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `scikit-learn` | Baseline ML models & text vectorisation |
| `torch>=2.0.0` | PyTorch LSTM deep-learning model |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Accuracy comparison charts |

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/SarahAbada/my-ai-backend.git
    cd my-ai-backend
    ```

2.  **Set up the environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

To train **all** models (Naive Bayes, SVM, Logistic Regression, and LSTM) and compare their accuracy:

```bash
python main.py
```

#### Running only the LSTM

```python
from lstm_model import train_lstm
import pandas as pd

data = pd.read_csv("data/reviews.csv")
model, accuracy, vectorizer = train_lstm(data["text"], data["label"])
print(f"LSTM Test Accuracy: {accuracy:.4f}")
```

*Note: The current version runs a training benchmark on startup. API endpoints are under active development.*

## 📊 Model Performance

| Model | Accuracy (Est.) |
|-------|-----------------|
| **Naive Bayes** | ~85% |
| **Linear SVM** | ~89% |
| **Logistic Regression** | ~88% |
| **LSTM (PyTorch)** | ~90%+ |

> Accuracy figures are estimates on the IMDb dataset. Run `python main.py` to generate an up-to-date `model_comparison.png` chart.

## 🔮 Future Improvements

-   [x] **Deep Learning Integration**: LSTM classifier implemented with PyTorch.
-   [ ] **Transformer Models**: Experimenting with BERT for improved context understanding.
-   [ ] **Dockerization**: Containerizing the application for easy cloud deployment.
-   [ ] **CI/CD Pipeline**: Automating testing and linting with GitHub Actions.
-   [ ] **User Frontend**: A simple React dashboard to visualize sentiment trends.

## 🤝 Contributing

Contributions are welcome! Please compile `requirements.txt` if adding new dependencies.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
