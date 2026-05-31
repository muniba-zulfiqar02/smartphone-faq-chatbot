from flask import Flask, request, jsonify, render_template
from chatbot import get_answer

app = Flask(__name__)

# ── Route 1: Home page ────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── Route 2: Chat endpoint ────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please type something!'})
    
    response = get_answer(user_message)
    return jsonify({'response': response})

# ── Run the app ───────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)