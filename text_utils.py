
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    cleaned = [
        ps.stem(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(cleaned)
print(transform_text("This is a sample message??!"))