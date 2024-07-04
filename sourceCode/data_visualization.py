import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from data_analysis2 import SentimentRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_history(history):
    train_loss, val_loss, train_acc, val_acc = history
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

def visualize_deep_model_results(model_path, test_data_path):
    model = SentimentRNN(vocab_size=1001, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    x_test_pad = np.load(test_data_path)
    y_test = np.load('y_test.npy')
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_pad, dtype=torch.long), torch.tensor(y_test, dtype=torch.float)),
        batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        output = model(inputs)
        preds = torch.round(output.squeeze()).detach().cpu().numpy()
        all_preds.extend(preds)

    sns.histplot(all_preds, bins=50, kde=False)
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Sentiments')
    plt.show()

def plot_class_distribution(y_train, y_val, y_test):
    sns.countplot(x=y_train)
    plt.title('Train Set Class Distribution')
    plt.show()

    sns.countplot(x=y_val)
    plt.title('Validation Set Class Distribution')
    plt.show()

    sns.countplot(x=y_test)
    plt.title('Test Set Class Distribution')
    plt.show()
