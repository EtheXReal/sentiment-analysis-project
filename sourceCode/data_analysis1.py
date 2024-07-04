import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train_classic_model():
    # Load the preprocessed data
    train_data = pd.read_csv('preprocessed_imdb_train.csv')
    val_data = pd.read_csv('preprocessed_imdb_val.csv')
    test_data = pd.read_csv('preprocessed_imdb_test.csv')

    X_train = train_data['preprocessed_text']
    y_train = train_data['sentiment']
    X_val = val_data['preprocessed_text']
    y_val = val_data['sentiment']
    X_test = test_data['preprocessed_text']
    y_test = test_data['sentiment']

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

if __name__ == '__main__':
    train_classic_model()
