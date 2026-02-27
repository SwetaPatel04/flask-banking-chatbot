# Import Flask to create the web server
from flask import Flask, request, jsonify

# Import CORS to allow frontend to talk to this API
from flask_cors import CORS

# Import pickle to load our saved model files
import pickle

# Import random to pick a random response from matching responses
import random

# Import nltk for text processing
import nltk

# Import word tokenizer to split sentences into words
from nltk.tokenize import word_tokenize

# Import stemmer to reduce words to root form
from nltk.stem import PorterStemmer

# Download required nltk data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Create Flask app instance
app = Flask(__name__)

# Enable CORS so frontend can talk to this API
CORS(app)

# Create stemmer instance
stemmer = PorterStemmer()

# Load the trained model from disk
with open("model/chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer — needed to process new messages
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the intents — needed to look up responses
with open("model/intents.pkl", "rb") as f:
    intents = pickle.load(f)

# Function to clean and normalize incoming messages
# Same cleaning we did during training so the model understands correctly
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into words
    tokens = word_tokenize(text)
    # Stem each word to root form
    tokens = [stemmer.stem(token) for token in tokens]
    # Join back to string
    return " ".join(tokens)

# Health check endpoint
# Example: GET http://localhost:5000/health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "flask-banking-chatbot"
    })

# Main chat endpoint — receives message and returns response
# Example: POST http://localhost:5000/chat
@app.route("/chat", methods=["POST"])
def chat():
    # Get the JSON data from the request
    data = request.get_json()

    # Validate that message field is present
    if not data or "message" not in data:
        return jsonify({
            "error": "Missing required field: message"
        }), 400

    # Get the user message
    message = data["message"].strip()

    # Validate message is not empty
    if not message:
        return jsonify({
            "error": "Message cannot be empty"
        }), 400

    # Validate message length — prevent abuse
    if len(message) > 500:
        return jsonify({
            "error": "Message too long — maximum 500 characters"
        }), 400

    # Clean the message using same process as training
    cleaned = clean_text(message)

    # Convert message to TF-IDF vector
    vector = vectorizer.transform([cleaned])

    # Get predicted intent tag
    predicted_tag = model.predict(vector)[0]

    # Get confidence probability for the prediction
    probabilities = model.predict_proba(vector)[0]
    confidence    = round(float(max(probabilities)), 4)

    # If confidence is too low the model is unsure — return fallback
    if confidence < 0.15:
        return jsonify({
            "message":    message,
            "intent":     "unknown",
            "confidence": confidence,
            "response":   "I'm not sure I understand. Could you rephrase that? You can also call TD at 1-866-222-3456 for assistance."
        })

    # Find the matching intent in our intents data
    response_text = "I'm sorry, I couldn't find an answer. Please call TD at 1-866-222-3456."
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            # Pick a random response from the matching intent
            response_text = random.choice(intent["responses"])
            break

    # Return the response with intent and confidence info
    return jsonify({
        "message":    message,
        "intent":     predicted_tag,
        "confidence": confidence,
        "response":   response_text
    })

# Endpoint to get all available intents
# Useful for the frontend to show suggested questions
# Example: GET http://localhost:5000/intents
@app.route("/intents", methods=["GET"])
def get_intents():
    # Return just the tags and first pattern for each intent
    intent_list = []
    for intent in intents["intents"]:
        intent_list.append({
            "tag":     intent["tag"],
            "example": intent["patterns"][0]
        })
    return jsonify({
        "intents": intent_list,
        "total":   len(intent_list)
    })

# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)