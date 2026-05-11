# ===============================
# IMPORT LIBRARY
# ===============================
from google_play_scraper import reviews, Sort
import pandas as pd
import numpy as np
import re
import string

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# DOWNLOAD NLTK RESOURCE
# ===============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ===============================
# SCRAPING 100 REVIEW
# ===============================
result, _ = reviews(
    'com.zhiliaoapp.musically',  # TikTok App ID
    lang='id',
    country='id',
    sort=Sort.MOST_RELEVANT,
    count=100
)

df = pd.DataFrame(result)[['content', 'score']]
df.columns = ['review', 'rating']

scraped_json = df.to_dict(orient='records')

with open('tiktok_reviews_raw.json', 'w', encoding='utf-8') as f:
    import json
    json.dump(scraped_json, f, ensure_ascii=False, indent=4)

print("Data scraping disimpan ke: tiktok_reviews_raw.json")

# ===============================
# LABEL SENTIMEN DARI RATING
# ===============================
def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(label_sentiment)

# Buang netral (opsional, tapi umum)
df = df[df['sentiment'] != 'neutral']

# ===============================
# PREPROCESSING TEXT
# ===============================
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

df['clean_review'] = df['review'].apply(preprocess)

# ===============================
# TF-IDF (Cocok untuk BernoulliNB)
# ===============================
vectorizer = TfidfVectorizer(binary=True)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# NAIVE BAYES (KODE KAMU)
# ===============================
naive_bayes = BernoulliNB()

naive_bayes.fit(X_train.toarray(), y_train)

y_pred_train_nb = naive_bayes.predict(X_train.toarray())
y_pred_test_nb = naive_bayes.predict(X_test.toarray())

accuracy_train_nb = accuracy_score(y_pred_train_nb, y_train)
accuracy_test_nb = accuracy_score(y_pred_test_nb, y_test)

print('Naive Bayes - accuracy_train:', accuracy_train_nb)
print('Naive Bayes - accuracy_test:', accuracy_test_nb)

# ===============================
# LAPORAN KLASIFIKASI
# ===============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test_nb))
