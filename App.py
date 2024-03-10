import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randrange
import random
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))



def stemming(news):
    stemmed_news = re.sub('^a-zA-Z', ' ', news)
    stemmed_news = stemmed_news.lower()
    stemmed_news = stemmed_news.split()
    stemmed_news = [port_stem.stem(word) for word in stemmed_news if not word in stopwords.words('english')]
    stemmed_news = ' '.join(stemmed_news)
    return stemmed_news

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


def get_random_news_text():
    

    # Load the news data from the CSV file
    news_data = pd.read_csv("test.csv")  
    news_data = news_data.dropna(how="any").reset_index(drop=True)

    # Select a random news text article
    random_index = random.randint(0, len(news_data) - 1)
    random_news_text = news_data.loc[random_index, "text"] 

    return random_news_text



if __name__ == '__main__':

  st.title('Fake News Classification app ')

  # Set an empty string as default content (optional)
  sentence = st.text_area("Enter your news content here OR Fetch Random News", "", height=300)

  # Use radio buttons to choose input method
  input_method = st.radio("Input Method:", ("Manual Input", "Fetch Random News"))

  # Update text box content based on chosen input method
  if input_method == "Fetch Random News":
      # Fetch random news and update the text box with a key argument
      sentence = st.text_area("Enter your news content here OR Fetch Random News", get_random_news_text(), height=300, key="news_text")
      st.write("**Fetched Random News:**") 

  # Rest of your code...
  predict_btt = st.button("Predict")


  if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.warning('Unreliable')
        if prediction_class == [1]:
            st.success('Reliable')