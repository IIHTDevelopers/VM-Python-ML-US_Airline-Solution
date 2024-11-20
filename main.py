import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# Define Functions

def load_data(file_path):
    """
    Load the dataset and select relevant columns.
    """
    df = pd.read_csv(file_path)
    df = df[['airline_sentiment', 'text', 'airline']]
    df = df.rename(columns={"airline_sentiment": "sentiment", "text": "tweet"})
    return df


def preprocess_data(df):
    """
    Preprocess the text data: cleaning, mapping sentiments, etc.
    """

    def clean_tweet(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)
    return df


def split_data(df):
    """
    Split the dataset into training and testing sets.
    """
    X = df['cleaned_tweet']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize_data(X_train, X_test):
    """
    Convert text data into TF-IDF vectors.
    """
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf


def train_and_save_model(X_train_tfidf, y_train, tfidf):
    """
    Train a Logistic Regression model and save it.
    """
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, 'sentiment_model.pkl')  # Save model
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')  # Save vectorizer
    return model


def load_model_and_vectorizer():
    """
    Load the saved model and vectorizer.
    """
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf


def predict_sentiments(model, tfidf, data):
    """
    Predict sentiments using the trained model.
    """
    data_tfidf = tfidf.transform(data['cleaned_tweet'])
    predictions = model.predict(data_tfidf)
    data['predicted_sentiment'] = predictions
    return data


# Analytical Functions

def sentiment_distribution(data):
    return data['predicted_sentiment'].value_counts().to_dict()


def most_active_airline(data):
    return data['airline'].value_counts().idxmax()


def percentage_negative_tweets(data):
    total_tweets = len(data)
    negative_tweets = len(data[data['predicted_sentiment'] == "negative"])
    return round((negative_tweets / total_tweets) * 100, 2)


def top_airlines_by_positive_sentiment(data):
    positive_counts = data[data['predicted_sentiment'] == "positive"]['airline'].value_counts()
    return positive_counts.to_dict()


def frequent_bigrams(data, sentiment):
    sentiment_text = data[data['predicted_sentiment'] == sentiment]['cleaned_tweet']
    if sentiment_text.empty:
        return []
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bigrams = vectorizer.fit_transform(sentiment_text)
    bigram_counts = zip(vectorizer.get_feature_names_out(), bigrams.toarray().sum(axis=0))
    return sorted(bigram_counts, key=lambda x: x[1], reverse=True)[:10]


def average_tweet_length(data):
    tweet_lengths = data['cleaned_tweet'].apply(lambda x: len(x.split()))
    return round(tweet_lengths.mean(), 2)


def most_negative_airline(data):
    negative_counts = data[data['predicted_sentiment'] == "negative"]['airline'].value_counts()
    total_counts = data['airline'].value_counts()
    negative_ratio = (negative_counts / total_counts).fillna(0)
    return negative_ratio.idxmax(), round(negative_ratio.max() * 100, 2)


def airline_with_highest_positive_ratio(data):
    positive_counts = data[data['predicted_sentiment'] == "positive"]['airline'].value_counts()
    total_counts = data['airline'].value_counts()
    positive_ratio = (positive_counts / total_counts).fillna(0)
    return positive_ratio.idxmax(), round(positive_ratio.max() * 100, 2)


def most_common_trigrams(data):
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    trigrams = vectorizer.fit_transform(data['cleaned_tweet'])
    trigram_counts = zip(vectorizer.get_feature_names_out(), trigrams.toarray().sum(axis=0))
    return sorted(trigram_counts, key=lambda x: x[1], reverse=True)[:10]


def top_airlines_by_neutral_sentiment(data):
    neutral_counts = data[data['predicted_sentiment'] == "neutral"]['airline'].value_counts()
    return neutral_counts.to_dict()


def most_frequent_single_words(data, sentiment):
    text = " ".join(data[data['predicted_sentiment'] == sentiment]['cleaned_tweet'])
    return Counter(text.split()).most_common(10)


# Main Execution

def main():
    file_path = "Tweets.csv"  # Replace with the actual dataset path
    df = load_data(file_path)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, tfidf = vectorize_data(X_train, X_test)

    train_and_save_model(X_train_tfidf, y_train, tfidf)
    model, tfidf = load_model_and_vectorizer()

    df = predict_sentiments(model, tfidf, df)

    print("1. Sentiment Distribution:", sentiment_distribution(df))
    print("2. Most Active Airline:", most_active_airline(df))
    print("3. Percentage of Negative Tweets:", percentage_negative_tweets(df))
    print("4. Top Airlines by Positive Sentiment:", top_airlines_by_positive_sentiment(df))
    print("5. Frequent Bigrams for Positive Sentiment:", frequent_bigrams(df, "positive"))
    print("6. Average Tweet Length:", average_tweet_length(df))
    print("7. Most Negative Airline:", most_negative_airline(df))
    print("8. Airline with Highest Positive Ratio:", airline_with_highest_positive_ratio(df))
    print("9. Most Common Trigrams:", most_common_trigrams(df))
    print("10. Top Airlines by Neutral Sentiment:", top_airlines_by_neutral_sentiment(df))
    print("11. Most Frequent Words for Negative Sentiment:", most_frequent_single_words(df, "negative"))


if __name__ == "__main__":
    main()
