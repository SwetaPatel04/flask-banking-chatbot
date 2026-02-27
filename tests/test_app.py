# Import pytest for testing
import pytest

# Import sys and os to help Python find our api folder
import sys
import os

# Add root project folder to Python path so we can import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our Flask app
from api.app import app

# Pytest fixture — creates test client before each test
@pytest.fixture
def client():
    # Enable testing mode
    app.config["TESTING"] = True
    return app.test_client()

# Test 1 — Health check returns 200 and correct service name
def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json["status"] == "ok"
    assert res.json["service"] == "flask-banking-chatbot"

# Test 2 — Chat endpoint returns 200 with valid message
def test_chat_valid_message(client):
    res = client.post("/chat", json={"message": "hello"})
    assert res.status_code == 200
    assert "response" in res.json
    assert "intent" in res.json
    assert "confidence" in res.json

# Test 3 — Branch hours intent is detected correctly
def test_branch_hours_intent(client):
    res = client.post("/chat", json={"message": "what are your branch hours"})
    assert res.status_code == 200
    assert res.json["intent"] == "branch_hours"

# Test 4 — Lost card intent is detected correctly
def test_lost_card_intent(client):
    res = client.post("/chat", json={"message": "I lost my card"})
    assert res.status_code == 200
    assert res.json["intent"] == "lost_card"

# Test 5 — Greeting intent is detected correctly
def test_greeting_intent(client):
    res = client.post("/chat", json={"message": "hello"})
    assert res.status_code == 200
    assert res.json["intent"] == "greeting"

# Test 6 — Missing message field returns 400
def test_missing_message(client):
    res = client.post("/chat", json={})
    assert res.status_code == 400
    assert "error" in res.json

# Test 7 — Empty message returns 400
def test_empty_message(client):
    res = client.post("/chat", json={"message": ""})
    assert res.status_code == 400
    assert "error" in res.json

# Test 8 — Message too long returns 400
def test_message_too_long(client):
    res = client.post("/chat", json={"message": "a" * 501})
    assert res.status_code == 400
    assert "error" in res.json

# Test 9 — Confidence score is between 0 and 1
def test_confidence_range(client):
    res = client.post("/chat", json={"message": "what are your hours"})
    assert 0.0 <= res.json["confidence"] <= 1.0

# Test 10 — Intents endpoint returns list of intents
def test_get_intents(client):
    res = client.get("/intents")
    assert res.status_code == 200
    assert "intents" in res.json
    assert "total" in res.json
    assert res.json["total"] > 0