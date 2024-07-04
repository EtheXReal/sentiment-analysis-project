import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.sigmoid(out)

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    model.to(device)
    valid_loss_min = np.Inf

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.round(output.squeeze())
            train_correct += (preds == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = criterion(output.squeeze(), labels.float())
                val_loss += loss.item()
                preds = torch.round(output.squeeze())
                val_correct += (preds == labels).sum().item()

        val_losses.append(val_loss / len(valid_loader))
        val_accuracies.append(val_correct / len(valid_loader.dataset))

        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Valid Loss: {val_losses[-1]}, Train Acc: {train_accuracies[-1]}, Valid Acc: {val_accuracies[-1]}')

        if val_losses[-1] <= valid_loss_min:
            torch.save(model.state_dict(), 'model2.pth')
            valid_loss_min = val_losses[-1]

    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    test_losses = []
    test_acc = 0.0
    criterion = nn.BCELoss()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output.squeeze(), labels.float())
        test_losses.append(loss.item())
        preds = torch.round(output.squeeze())
        test_acc += (preds == labels).sum().item()

    test_loss = np.mean(test_losses)
    test_acc = test_acc / len(test_loader.dataset)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
