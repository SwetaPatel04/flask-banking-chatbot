# Import json to load our intents.json file
import json

# Import pickle to save the trained model and vectorizer
import pickle

# Import numpy for array operations
import numpy as np

# Import random to pick random responses later
import random

# Import nltk for text processing
import nltk

# Import the word tokenizer — splits sentences into individual words
from nltk.tokenize import word_tokenize

# Import the stemmer — reduces words to their root form
# e.g. "running" -> "run", "hours" -> "hour"
from nltk.stem import PorterStemmer

# Import TF-IDF vectorizer — converts text into numbers the model understands
# TF-IDF = Term Frequency Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Naive Bayes classifier — works great for text classification
from sklearn.naive_bayes import MultinomialNB

# Import accuracy score to evaluate our model
from sklearn.metrics import accuracy_score

# Import train test split to evaluate the model
from sklearn.model_selection import train_test_split

# Download required NLTK data files (only needed once)
# punkt = tokenizer, stopwords = common words to ignore
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# Create a stemmer instance to reduce words to root form
stemmer = PorterStemmer()

# Load the intents JSON file we created
with open("data/intents.json", "r") as f:
    intents = json.load(f)

# Lists to store training data
# sentences = all the pattern sentences from intents.json
# labels = the matching intent tag for each sentence
sentences = []
labels    = []

# Loop through each intent and collect all patterns and their tags
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Add the pattern sentence to our sentences list
        sentences.append(pattern)
        # Add the matching tag as the label for this sentence
        labels.append(intent["tag"])

# Function to clean and normalize text
# This ensures "What are YOUR hours?" becomes "what are your hours"
def clean_text(text):
    # Convert to lowercase so "Hello" and "hello" are treated the same
    text = text.lower()
    # Tokenize — split into individual words
    tokens = word_tokenize(text)
    # Stem each word — reduce to root form
    tokens = [stemmer.stem(token) for token in tokens]
    # Join back into a single string
    return " ".join(tokens)

# Clean all sentences in our training data
cleaned_sentences = [clean_text(s) for s in sentences]

# Create TF-IDF vectorizer
# This converts our text sentences into numerical vectors
# The model can only understand numbers, not words
vectorizer = TfidfVectorizer()

# Fit the vectorizer on our sentences and transform them into vectors
X = vectorizer.fit_transform(cleaned_sentences)

# Convert labels list to numpy array
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Naive Bayes classifier
# Naive Bayes works well for text because it considers each word independently
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model on test data
predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.1f}%")

# Save the trained model to a file so the API can load it
with open("model/chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the vectorizer too — we need it to process new messages
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the intents so the API can look up responses
with open("model/intents.pkl", "wb") as f:
    pickle.dump(intents, f)

# Confirm everything was saved
print("Model saved to model/chatbot_model.pkl")
print("Vectorizer saved to model/vectorizer.pkl")
print("Intents saved to model/intents.pkl")
