import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def load_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_text(df, text_column):
    def preprocess(text):
        text = re.sub(r'<.*?>', '', text)
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    df['preprocessed_text'] = df[text_column].apply(preprocess)
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return df

def tokenize_and_pad(x, vocab, seq_len):
    tokenized_data = [[vocab[word] for word in sentence.split() if word in vocab] for sentence in x]
    padded_data = np.zeros((len(tokenized_data), seq_len), dtype=int)
    for i, sentence in enumerate(tokenized_data):
        if len(sentence) != 0:
            padded_data[i, -len(sentence):] = np.array(sentence)[:seq_len]
    return padded_data

def split_data(df, vocab_size=1000, seq_len=500):
    X = df['preprocessed_text'].values
    y = df['sentiment'].values
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    word_list = []
    for sent in x_train:
        for word in sent.split():
            word_list.append(word)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:vocab_size]
    vocab = {w: i+1 for i, w in enumerate(corpus_)}

    x_train_pad = tokenize_and_pad(x_train, vocab, seq_len)
    x_val_pad = tokenize_and_pad(x_val, vocab, seq_len)
    x_test_pad = tokenize_and_pad(x_test, vocab, seq_len)

    return x_train_pad, x_val_pad, x_test_pad, y_train, y_val, y_test, vocab

if __name__ == "__main__":
    df = load_dataset('imdb_dataset.csv')
    df = preprocess_text(df, 'review')
    x_train_pad, x_val_pad, x_test_pad, y_train, y_val, y_test, vocab = split_data(df)
    np.save('x_train_pad.npy', x_train_pad)
    np.save('x_val_pad.npy', x_val_pad)
    np.save('x_test_pad.npy', x_test_pad)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    np.save('vocab.npy', vocab)
