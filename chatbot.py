import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (only runs once)
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# ── Step 1: Load FAQ data ──────────────────────────────
with open('faq_data.json', 'r') as f:
    data = json.load(f)

faqs = data['faqs']
questions = [faq['question'] for faq in faqs]
answers   = [faq['answer']   for faq in faqs]

# ── Step 2: Preprocess text ───────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase everything
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize (split into words)
    tokens = nltk.word_tokenize(text)
    # Remove stopwords (like "is", "the", "a")
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

clean_questions = [preprocess(q) for q in questions]

# ── Step 3: TF-IDF Vectorizer ─────────────────────────
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_questions)

# ── Step 4: Match user question ───────────────────────
def get_answer(user_input):
    clean_input = preprocess(user_input)
    input_vector = vectorizer.transform([clean_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.1:
        return "Sorry, I couldn't find an answer to that. Please try rephrasing your question."
    
    return answers[best_match_index]