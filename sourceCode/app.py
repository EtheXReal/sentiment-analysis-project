import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from data_preprocessing import preprocess_text, tokenize_and_pad
from data_analysis2 import SentimentRNN

app = Flask(__name__)

# Load the trained model and other required data
model = SentimentRNN(vocab_size=1001, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, dropout=0.5)
model.load_state_dict(torch.load('model2.pth'))
model.eval()
vocab = np.load('vocab.npy', allow_pickle=True).item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_input(text):
    df = pd.DataFrame([text], columns=['review'])
    df = preprocess_text(df, 'review')
    x = tokenize_and_pad(df['preprocessed_text'].values, vocab, seq_len=500)
    return torch.tensor(x, dtype=torch.long).to(device)

def predict_sentiment(text):
    input_data = preprocess_input(text)
    with torch.no_grad():
        output = model(input_data)
        probability = output.item()
    return probability

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    probability = predict_sentiment(text)
    sentiment = "positive" if probability > 0.5 else "negative"
    probability = probability if sentiment == "positive" else (1 - probability)
    response = {
        'predicted_sentiment': sentiment,
        'probability': probability
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
