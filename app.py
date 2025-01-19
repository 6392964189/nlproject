import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load and preprocess the dataset
def load_data():
    data = pd.read_csv(r"C:\Users\HP\Downloads\amazon_alexa.tsv", sep='\t')
    data = data[['verified_reviews', 'feedback']].dropna()
    return data

# Train the model
def train_model(data):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = data['verified_reviews']
    y = data['feedback']
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(random_state=42)
    model.fit(X_vec, y)

    return vectorizer, model

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("This app predicts whether a review has Positive or Negative sentiment.")

    # Load data and train model
    data = load_data()
    vectorizer, model = train_model(data)

    # User input
    user_input = st.text_area("Enter a review:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            user_vec = vectorizer.transform([user_input])
            prediction = model.predict(user_vec)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            st.write(f"### Sentiment: {sentiment}")
        else:
            st.write("Please enter a valid review.")

# Run the app
if __name__ == "__main__":
    main()
