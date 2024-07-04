# Sentiment Analysis of Movie Reviews

This project performs sentiment analysis on IMDB movie reviews using both traditional machine learning (Logistic Regression) and deep learning (LSTM) approaches.

## Project Structure

- `data_preprocessing.py`: Preprocesses the raw IMDB dataset
- `data_analysis1.py`: Implements Logistic Regression model
- `data_analysis2.py`: Implements LSTM model
- `data_visualization.py`: Creates visualizations of the results
- `app.py`: Flask application for real-time sentiment prediction
- `model1.pickle`: Saved Logistic Regression model
- `model2.pth`: Saved LSTM model
- `vectorizer.pickle`: Saved TF-IDF vectorizer
- `vocab.npy`: Vocabulary for the LSTM model

## Setup and Installation

1. Clone this repository
2. Install required packages: pip install -r requirements.txt
3. Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and rename it to "imdb_dataset.csv"
4. Place the dataset in the project root directory

## Usage

1. Run data preprocessing:python data_preprocessing.py
2. Train and evaluate models:
python data_analysis1.py
python data_analysis2.py
3. Visualize results:
python data_visualization.py
4. Run the Flask app:
python app.py


## Contributors

- Shilong Luo, Yuxuan Liu

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.