import streamlit as st
import os
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "twitter_analysis_model.sav"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

stop_words = set(stopwords.words("english"))
port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1DA1F2;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #333333;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #1DA1F2;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0d8ddb;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 16px;
    }
    .footer a {
        color: #1DA1F2;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🐦 Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Check if a tweet is Positive 😊 or Negative 😡</p>", unsafe_allow_html=True)

tweet_input = st.text_area("✍️ Type your tweet here:")

if st.button("Predict Sentiment"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a tweet first!")
    else:
        processed_tweet = stemming(tweet_input)
        vectorized_tweet = vectorizer.transform([processed_tweet])
        prediction = model.predict(vectorized_tweet)[0]

        if prediction == 1:
            st.markdown("<div class='result-card positive'>✅ Positive Sentiment 😊</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card negative'>❌ Negative Sentiment 😡</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("🎲 Try Random Tweets")

examples = [
    "Absolutely loving the new phone update! Battery lasts forever 🔋💯",
    "This app keeps crashing, so frustrating 😡👎",
    "Going to the gym later, need to stay consistent 🏋️‍♂️",
    "Best movie I’ve seen this year! 🎬🔥",
    "Worst customer service experience ever. Never again 😤"
]

for example in examples:
    if st.button(example):
        processed = stemming(example)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.markdown(f"<div class='result-card positive'>✅ {example}<br>Sentiment: Positive 😊</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card negative'>❌ {example}<br>Sentiment: Negative 😡</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p class='footer'>📂 Check the full project on "
    "<a href='https://github.com/wigjatin/Twitter-Sentiment-Analysis' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)
