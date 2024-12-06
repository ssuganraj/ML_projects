import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib

# Load the dataset
data = pd.read_csv('imdb_top_1000.csv')

# Handle missing values
data['Certificate'] = data['Certificate'].fillna('Not Available')
data['Meta_score'] = data['Meta_score'].fillna(data['Meta_score'].median())
data['Gross'] = data['Gross'].replace({',': ''}, regex=True)
data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')
data['Gross'] = data['Gross'].fillna(data['Gross'].median())

# Ensure required columns are present
required_columns = ['Overview', 'IMDB_Rating']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the dataset: {missing_columns}")

# Download NLTK stopwords (only if not downloaded already)
nltk.download('stopwords', quiet=True)

# Preprocessing function
def preprocess_text(text):
    if pd.isnull(text):  # Handle NaN values
        return ""
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Preprocess text
data['Overview'] = data['Overview'].fillna('')  # Replace NaN with empty string
data['cleaned_review'] = data['Overview'].apply(preprocess_text)

# Create the 'sentiment' column
data['IMDB_Rating'] = pd.to_numeric(data['IMDB_Rating'], errors='coerce')
data = data.dropna(subset=['IMDB_Rating'])
data['sentiment'] = data['IMDB_Rating'].apply(lambda x: 'positive' if x >= 8.5 else 'negative')

# Balance the dataset using oversampling
positive_samples = data[data['sentiment'] == 'positive']
negative_samples = data[data['sentiment'] == 'negative']
positive_oversampled = resample(
    positive_samples,
    replace=True,  # Oversample with replacement
    n_samples=len(negative_samples),  # Match the count of negative samples
    random_state=42
)
balanced_data = pd.concat([negative_samples, positive_oversampled])
print("Balanced Sentiment Distribution:\n", balanced_data['sentiment'].value_counts())

# Split data
X = balanced_data['cleaned_review']
y = balanced_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Check the distribution of sentiments
print("Sentiment Distribution after Balancing:")
print(balanced_data['sentiment'].value_counts())
