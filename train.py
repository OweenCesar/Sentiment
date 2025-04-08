import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.preprocessing import load_data, build_vocab, text_to_tensor
from tqdm import tqdm

# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 5

# Load data
X_train, X_test, y_train, y_test = load_data("data/IMDB Dataset.csv")
vocab = build_vocab(X_train)

# Convert to tensors
X_train_tensor = text_to_tensor(X_train, vocab)
X_test_tensor = text_to_tensor(X_test, vocab)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Model definition
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return torch.sigmoid(self.fc(hidden[-1]))

# Training loop
model = SentimentLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch_X, batch_y in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "models/sentiment_lstm.pt")

import pickle
with open('models/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)  # Save alongside the model 