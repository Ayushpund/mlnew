from flask import Flask, request, jsonify
import pickle
from googletrans import Translator

from rapidfuzz import fuzz

# Initialize Flask app
app = Flask(__name__)

# Load the FAQ data
faq_file = 'faq_data.pkl'
try:
    with open(faq_file, 'rb') as f:
        faq_data = pickle.load(f)
    print("FAQ data loaded successfully!")
except Exception as e:
    print(f"Error loading FAQ data: {e}")
    exit(1)

# Initialize translator
translator = Translator()

def translate_text(text, src_language, dest_language):
    """Translate text between source and destination languages."""
    try:
        translated = translator.translate(text, src=src_language, dest=dest_language)
        return translated.text

    except Exception as e:
        return f"An unexpected error occurred during translation: {str(e)}"

def find_answer(user_query, faq_data):
    """Find the most relevant answer based on 70% similarity using fuzzy matching."""
    best_match = None
    highest_score = 0
    threshold = 70  # Set threshold to 70%

    for row in faq_data:
        score = fuzz.partial_ratio(user_query.lower(), row['Question'].lower())
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = row['Answer']

    if best_match:
        return best_match
    return "Sorry, I couldn't find an answer to your query."

@app.route('/')
def index():
    return "Welcome to the Chatbot API! Use /chat endpoint to interact with the chatbot."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Extract user input and language from form data
        user_query = request.form.get('query')
        user_language = request.form.get('language', 'en')

        # Validate inputs
        if not user_query:
            return jsonify({"error": "Query is required."}), 400

        # Translate user query to English
        query_in_english = translate_text(user_query, user_language, 'en')
        if "Translation failed" in query_in_english or "An unexpected error occurred" in query_in_english:
            return jsonify({"error": query_in_english}), 500

        # Find the answer in the FAQ data
        answer_in_english = find_answer(query_in_english, faq_data)

        # Translate the answer back to the user's language
        answer_in_user_language = translate_text(answer_in_english, 'en', user_language)
        if "Translation failed" in answer_in_user_language or "An unexpected error occurred" in answer_in_user_language:
            return jsonify({"error": answer_in_user_language}), 500

        # Return the response
        return jsonify({
            "query": user_query,
            "language": user_language,
            "answer": answer_in_user_language
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
