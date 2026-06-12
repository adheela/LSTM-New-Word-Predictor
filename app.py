import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



# loading the tensorflow model for prediction
model = load_model('nextword_model.h5')

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

max_len = 214
reverse_index = {idx:word for word,idx in tokenizer.word_index.items()}

def generate_text(seed_text, num_words):
  text = seed_text
  for _ in range(num_words):
    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([seq], maxlen=max_len, padding='pre')
    preds = model.predict(padded, verbose=0)
    pos = np.argmax(preds)
    next_word = reverse_index(pos," ")
    text +=" " + next_word
  return text




st.title('Next Word Prediction with Deep Learning')

seed = st.text_input('Enter a starting text: ','Hello')

num_words = st.slider('Number of words to generate',1,20,10)

Predict = st.button("Generate")



if Predict :
    result = generate_text(seed, num_words)
    st.write(result)


