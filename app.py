import pickle
import torch
import torch.nn as nn
import streamlit as st
from utils.preprocessing import clean_text, text_to_tensor

# Load vocabulary
with open('models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Model definition MUST match training exactly
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return torch.sigmoid(self.fc(hidden[-1]))

# Initialize model WITH CORRECT VOCAB SIZE
model = SentimentLSTM(len(vocab))  # Critical change!
model.load_state_dict(torch.load("models/sentiment_lstm.pt", map_location='cpu'))
model.eval()

 
# Streamlit UI
st.title("IMDB Sentiment Analysis")
review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if review:
        cleaned = clean_text(review)
        tensor = text_to_tensor([cleaned], vocab)
        with torch.no_grad():
            pred = model(tensor).item()
        st.success(f"Positive: {pred:.2f}" if pred > 0.5 else f"Negative: {1-pred:.2f}")