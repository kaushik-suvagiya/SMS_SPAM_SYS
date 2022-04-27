import streamlit as st
import pickle
#import requests
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title("Email or SMS Spam Classification")
input_sms = st.text_area("Enter the Mail or SMS")
#preprocessing of input
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if st.button('Prediction'):

          tranformed_sms = transform_text(input_sms)
          vector_input = tfidf.transform([tranformed_sms])  # vactorise
          result = model.predict(vector_input)[0]  # Prediction
          if result == 1:
             st.subheader("The mail or SMS is spam")
          else:
             st.subheader("The mail or SMS is not spam")
#else:
          #print("Enter the appropriate input")
if st.button('Clear Results'):
    #input_sms=st.empty()
    #input_sms.empty()
    st.experimental_memo.clear()


