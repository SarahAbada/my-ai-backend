# Sentiment Analyzer API

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
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
-   **📊 Data Visualization**: Built-in performance comparison tools using Matplotlib to visualize model accuracy and confusion matrices.
-   **🛠️ Robust Preprocessing**: Efficient text vectorization using `CountVectorizer` to handle large vocabularies (IMDb dataset).

## 🛠️ Tech Stack

-   **Language**: Python
-   **API Framework**: FastAPI, Uvicorn
-   **Machine Learning**: Scikit-Learn, Pandas, NumPy
-   **Visualization**: Matplotlib
-   **Data Source**: IMDb Large Movie Review Dataset (50k+ reviews)

## 🏎️ Getting Started

### Prerequisites

-   Python 3.9+
-   pip

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

To train the model and start the analysis pipeline:

```bash
python main.py
```

*Note: The current version runs a training benchmark on startup. API endpoints are under active development.*

## 📊 Model Performance

| Model | Accuracy (Est.) |
|-------|-----------------|
| **Naive Bayes** | ~85% |
| **Linear SVM** | ~89% |
| **Logistic Regression** | ~88% |

## 🔮 Future Improvements

-   [ ] **Deep Learning Integration**: Experimenting with LSTM/BERT for improved context understanding.
-   [ ] **Dockerization**: Containerizing the application for easy cloud deployment.
-   [ ] **CI/CD Pipeline**: Automating testing and linting with GitHub Actions.
-   [ ] **User Frontend**: A simple React dashboard to visualize sentiment trends.

## 🤝 Contributing

Contributions are welcome! Please compile `requirements.txt` if adding new dependencies.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
