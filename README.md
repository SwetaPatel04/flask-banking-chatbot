# 🤖 Flask Banking Chatbot API

An intent-based banking chatbot API built with Flask, scikit-learn, and NLTK.
Classifies user banking questions and returns smart responses with confidence 
scoring and an interactive chat interface.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-10%20Passing-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project simulates the backend brain of a banking virtual assistant.
It receives natural language questions, classifies the intent using a 
Naive Bayes classifier, and returns appropriate banking responses with 
a confidence score.

Built to demonstrate:
- NLP text processing with NLTK (tokenization, stemming)
- TF-IDF vectorization for converting text to numbers
- Intent classification with Naive Bayes (scikit-learn)
- Flask REST API design with input validation
- Interactive chat UI with confidence scoring
- Comprehensive unit testing with pytest (10 passing)

---

## 🗂️ Project Structure
```
flask-banking-chatbot/
├── api/
│   └── app.py              # Flask API — receives message, returns response
├── model/
│   ├── train_model.py      # Trains intent classifier
│   ├── chatbot_model.pkl   # Saved trained model
│   ├── vectorizer.pkl      # Saved TF-IDF vectorizer
│   └── intents.pkl         # Saved intents data
├── data/
│   └── intents.json        # All intents, patterns and responses
├── static/
│   └── index.html          # Interactive chat interface
├── tests/
│   └── test_app.py         # Unit tests (10 passing)
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/SwetaPatel04/flask-banking-chatbot.git
cd flask-banking-chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python model/train_model.py
```

### 5. Start the API
```bash
python api/app.py
```

### 6. Open the chat interface
Open `static/index.html` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/chat` | Send message, get response |
| GET | `/intents` | List all available intents |

### Example Request
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what are your branch hours?"}'
```

### Example Response
```json
{
  "message": "what are your branch hours?",
  "intent": "branch_hours",
  "confidence": 0.42,
  "response": "Most branches are open Monday to Friday 9am-5pm..."
}
```

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

Expected output:
```
test_health_check          PASSED ✅
test_chat_valid_message    PASSED ✅
test_branch_hours_intent   PASSED ✅
test_lost_card_intent      PASSED ✅
test_greeting_intent       PASSED ✅
test_missing_message       PASSED ✅
test_empty_message         PASSED ✅
test_message_too_long      PASSED ✅
test_confidence_range      PASSED ✅
test_get_intents           PASSED ✅

10 passed
```

---

## 💬 Supported Intents

| Intent | Example Question |
|--------|-----------------|
| greeting | "Hello", "Hi there" |
| branch_hours | "What are your branch hours?" |
| lost_card | "I lost my card" |
| account_balance | "What is my balance?" |
| transfer_money | "How do I send money?" |
| reset_pin | "I forgot my PIN" |
| interest_rates | "What is the interest rate?" |
| open_account | "How do I open an account?" |
| technical_support | "App not working" |
| thanks | "Thank you" |
| goodbye | "Bye" |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **NLP:** NLTK (tokenization, stemming), TF-IDF vectorization
- **ML:** scikit-learn (Naive Bayes classifier)
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Testing:** pytest (10 tests)
- **Dev Tools:** Git, VS Code, Thunder Client

---

## 👩‍💻 Author

**Sweta Patel** — Software Engineer | Python Developer | AI/ML Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sweta--patel-blue?style=flat-square)](https://linkedin.com/in/sweta-patel)
[![GitHub](https://img.shields.io/badge/GitHub-SwetaPatel04-black?style=flat-square)](https://github.com/SwetaPatel04)