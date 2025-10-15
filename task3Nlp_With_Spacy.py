import spacy

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews (replace with dataset samples)
reviews = [
    "I love the new Apple iPhone 13! The camera quality is superb.",
    "The Samsung Galaxy is terrible, it crashed after two days.",
    "Bought the Bose headphones, sound is amazing and very comfortable.",
    "The battery life of this Sony TV is disappointing.",
    "Amazon Echo Dot works great and has excellent voice recognition."
]

# Define simple positive and negative keyword lists for rule-based sentiment analysis
positive_keywords = ["love", "amazing", "superb", "great", "excellent", "comfortable"]
negative_keywords = ["terrible", "crashed", "disappointing", "bad", "poor", "slow"]

def analyze_sentiment(text):
    text_lower = text.lower()
    if any(pos_word in text_lower for pos_word in positive_keywords):
        return "Positive"
    elif any(neg_word in text_lower for neg_word in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

# Process each review, extract named entities and analyze sentiment
for review in reviews:
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = analyze_sentiment(review)

    print(f"Review: {review}")
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment}")
    print("=" * 40)
