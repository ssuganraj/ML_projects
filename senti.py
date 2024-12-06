import joblib
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('sentimental_model.pkl')
vectorizer = joblib.load('tfidf_vector.pkl')

print("Model and vectorizer loaded successfully.")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to predict sentiment
def predict_sentiment(review):
    # Preprocess the new review
    review_cleaned = preprocess_text(review)
    
    # Transform the review using the loaded vectorizer
    review_tfidf = vectorizer.transform([review_cleaned])
    
    # Predict sentiment using the loaded model
    sentiment = model.predict(review_tfidf)
    return sentiment[0]

# Example: Test with a new review
new_review = "This movie is fantastic and very entertaining!"
predicted_sentiment = predict_sentiment(new_review)
print(f"Predicted Sentiment: {predicted_sentiment}")
