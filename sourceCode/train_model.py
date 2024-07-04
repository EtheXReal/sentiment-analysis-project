import data_preprocessing as dp
import data_analysis2 as da2
import data_visualization as dv
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch.nn as nn


def train_classic_model():
    # Load the preprocessed data
    x_train_pad = np.load('x_train_pad.npy')
    x_val_pad = np.load('x_val_pad.npy')
    x_test_pad = np.load('x_test_pad.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    # Convert to 1D strings for TF-IDF
    X_train = [' '.join(map(str, review)) for review in x_train_pad]
    X_val = [' '.join(map(str, review)) for review in x_val_pad]
    X_test = [' '.join(map(str, review)) for review in x_test_pad]

    # Convert data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate the model on validation set
    y_pred = clf.predict(X_val_tfidf)
    val_report = classification_report(y_val, y_pred)
    val_accuracy = accuracy_score(y_val, y_pred)
    print("Validation Report:")
    print(val_report)
    print(f"Validation Accuracy: {val_accuracy}")

    # Evaluate the model on test set
    y_pred = clf.predict(X_test_tfidf)
    test_report = classification_report(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Report:")
    print(test_report)
    print(f"Test Accuracy: {test_accuracy}")

    # Save the trained model and vectorizer
    with open('model1.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)


def train_model(file_path):
    # 预处理数据
    df = dp.load_dataset(file_path)
    df = dp.preprocess_text(df, 'review')
    x_train_pad, x_val_pad, x_test_pad, y_train, y_val, y_test, vocab = dp.split_data(df)

    # 保存预处理后的数据
    np.save('x_train_pad.npy', x_train_pad)
    np.save('x_val_pad.npy', x_val_pad)
    np.save('x_test_pad.npy', x_test_pad)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    np.save('vocab.npy', vocab)

    # 加载数据
    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_pad, dtype=torch.long), torch.tensor(y_train, dtype=torch.float)),
        batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_pad, dtype=torch.long), torch.tensor(y_val, dtype=torch.float)),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_pad, dtype=torch.long), torch.tensor(y_test, dtype=torch.float)),
        batch_size=batch_size, shuffle=False)

    # 设置模型参数
    vocab_size = len(vocab) + 1
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    dropout = 0.5
    lr = 0.001
    epochs = 5

    # 初始化和训练模型
    model = da2.SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss, val_loss, train_acc, val_acc = da2.train_model(model, train_loader, valid_loader, criterion, optimizer, epochs)

    # 评估模型
    da2.evaluate_model(model, test_loader)

    # 可视化模型结果
    dv.plot_training_history((train_loss, val_loss, train_acc, val_acc))
    dv.visualize_deep_model_results('model2.pth', 'x_test_pad.npy')


if __name__ == "__main__":
    train_classic_model()  # 训练经典分类模型
    train_model('imdb_dataset.csv')  # 训练深度学习模型
