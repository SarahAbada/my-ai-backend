"""
PyTorch LSTM model for text sentiment classification.

Architecture
------------
- Embedding layer → LSTM (batch_first=True) → Linear classifier
- Many-to-one: only the last hidden state is forwarded to the Linear layer.

Training utilities
------------------
- train_lstm  : runs the training loop with Adam + CrossEntropyLoss
- evaluate_lstm: returns accuracy on any DataLoader
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """LSTM-based binary / multi-class text classifier.

    Parameters
    ----------
    input_size  : int   – number of input features (vocabulary size after
                          vectorisation)
    hidden_size : int   – number of hidden units in each LSTM cell
    num_layers  : int   – number of stacked LSTM layers
    num_classes : int   – number of output classes
    dropout     : float – dropout probability applied between LSTM layers
                          (ignored when num_layers == 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq, feature)
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len=1, input_size)
        out, _ = self.lstm(x)
        # Many-to-one: take the last time-step output
        out = out[:, -1, :]            # (batch, hidden_size)
        return self.fc(out)            # (batch, num_classes)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    texts,
    labels,
    vectorizer: CountVectorizer,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
):
    """Vectorise texts, perform 80/20 split, return (train_loader, test_loader).

    Inputs are reshaped to 3-D (samples, time_steps=1, features) as required
    by the LSTM's batch_first=True convention.
    """
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    X_train_vec = vectorizer.fit_transform(X_train_raw).toarray()
    X_test_vec = vectorizer.transform(X_test_raw).toarray()

    # Convert to FloatTensor and reshape to (samples, 1, features)
    X_train_t = torch.FloatTensor(X_train_vec).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_vec).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train_raw.values.copy())
    y_test_t = torch.LongTensor(y_test_raw.values.copy())

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lstm(
    texts,
    labels,
    num_epochs: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_grad_norm: float = 1.0,
    random_state: int = 42,
):
    """Train the LSTM classifier and return (model, test_accuracy, vectorizer).

    Parameters
    ----------
    texts        : array-like of raw text strings
    labels       : array-like of integer class labels
    num_epochs   : number of full passes over the training data
    hidden_size  : LSTM hidden dimension
    num_layers   : number of stacked LSTM layers
    dropout      : dropout between LSTM layers (ignored for single-layer)
    learning_rate: Adam learning rate
    batch_size   : mini-batch size for DataLoader
    max_grad_norm: clip_grad_norm_ threshold (prevents exploding gradients)
    random_state : seed for reproducible 80/20 split
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Using device: {device}")

    num_classes = len(set(int(y) for y in labels))

    vectorizer = CountVectorizer()
    train_loader, test_loader = build_dataloaders(
        texts, labels, vectorizer,
        test_size=0.2,
        batch_size=batch_size,
        random_state=random_state,
    )

    input_size = train_loader.dataset.tensors[0].shape[2]

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()                          # reset gradients
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(               # prevent exploding gradients
                model.parameters(), max_grad_norm
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[LSTM] Epoch [{epoch}/{num_epochs}]  Loss: {avg_loss:.4f}")

    test_acc = evaluate_lstm(model, test_loader, device)
    print(f"[LSTM] Test Accuracy: {test_acc:.4f}")
    return model, test_acc, vectorizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(
    model: LSTMClassifier,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return the accuracy of *model* on *data_loader*."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total if total > 0 else 0.0
