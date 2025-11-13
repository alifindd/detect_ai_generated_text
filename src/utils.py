import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import sent_tokenize
import re
import string
nltk.download("stopwords")
from nltk.corpus import stopwords

def load_data(path: str):
    return pd.read_csv(path)

def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black'
    ).generate(str(text))

    plt.figure(figsize=(10,5), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

# def avg_sentence_length(text: str):
#     # Splitting sentence
#     sentences = re.split(r'[.!?]+', text)

#     # Remove white-space after splitting
#     sentences = [s.strip() for s in sentences if s.strip()]

#     # 0 value checking
#     if len(sentences) == 0:
#       return 0

#     # Length of words per-sentence
#     word_counts = [len(s.split()) for s in sentences]

#     # Average sentence length
#     avg_len = sum(word_counts) / len(word_counts)
#     return word_counts

def avg_sentence_length(text: str, max_word=30):
    # Splitting sentence
    sentences = sent_tokenize(text)

    # Remove white-space after splitting
    sentences = [s.strip() for s in sentences if s.strip()]

    # 0 value checking
    if len(sentences) == 0:
      return 0
    elif len(sentences) <= 3:
      words = text.split()
      sentences = [" ".join(words[i:i+max_word])
      for i in range(0, len(words), max_word)]

    # Length of words per-sentence
    word_counts = [len(s.split()) for s in sentences]

    # Average sentence length
    avg_len = sum(word_counts) / len(word_counts)
    return avg_len

def word_length(text):
    return len(text.split())

def punctuation_ratio(text: str):
  extracted_punct = ""
  for char in text:
    if char in string.punctuation:
      extracted_punct += char

  word_counts = len(text.split())
  punct_ratio = len(extracted_punct)/word_counts

  return punct_ratio

def stopword_ratio(text: str):
  en_stopwords = stopwords.words("english")
  text_split = text.split()
  stop_words = []

  for word in text_split:
    if word in en_stopwords:
      stop_words.append(word)

  word_len = len(text_split)
  stop_ratio = len(stop_words) / word_len
  return stop_ratio

def find_ngram(df:pd.DataFrame, ngram:tuple[int,int]=(2,2), max_features:int=10):
  from sklearn.feature_extraction.text import CountVectorizer

  vectorizer = CountVectorizer(ngram_range=ngram, max_features=max_features)
  X = vectorizer.fit_transform(df)

  feature_names = vectorizer.get_feature_names_out()
  freq_ngram = X.sum(axis=0)

  ngram_dict = {}
  for i in range(len(feature_names)):
    ngram_dict.update({feature_names[i] : freq_ngram[0,i]})

  ngram_dict = sorted(ngram_dict.items(), key=lambda item: item[1], reverse=True)
  df_ngram = pd.DataFrame(ngram_dict, columns=["text", "count"])
  return df_ngram

  