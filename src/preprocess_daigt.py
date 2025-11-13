import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download("punkt_tab")
import string
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer



class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, return_token=False, train_data=False):
        self.return_token = return_token
        self.train_data = train_data

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        # Pastikan input string valid
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove escape characters
        text = re.sub(r"[\n\r\t\b]", " ", text)
        text = re.sub(r"[\\']", "'", text)
        # Remove links
        pat_link = r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*'
        text = re.sub(pat_link, "", text)
        # Normalize punctuation
        text = re.sub(r"[‘’]", "'", text)
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r"(…)", "...", text)
        text = re.sub(r"[–—−]", "-", text)
        text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
        # Tokenization
        tokens = word_tokenize(text)
        # Detokenize
        detok = TreebankWordDetokenizer()
        sentence = detok.detokenize(tokens)
        sentence = re.sub(r"\s+([?.!,;:])", r"\1", sentence)

        return tokens if self.return_token else sentence

    def transform(self, X):
        # Terima DataFrame, salin biar aman
        df = X.copy()
        # Pastikan ada kolom text
        if "text" not in df.columns:
            raise ValueError("DataFrame harus memiliki kolom 'text'")
        # Apply cleaning ke setiap baris
        df["text"] = df["text"].apply(self.clean_text)
        return df["text"] if self.train_data else df

class FeatureExtractor:
  def __init__(self):
    pass

  def word_length(self, text:str):
    return len(text.split()) if isinstance(text, str) else 0
  
  def avg_sentence_length(self, text: str, max_word=30):
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
    return avg_len if sentences else 0

  def punctuation_ratio(self, text: str):
    extracted_punct = ""
    for char in text:
      if char in string.punctuation:
        extracted_punct += char

    word_counts = len(text.split())
    punct_ratio = len(extracted_punct)/word_counts

    return punct_ratio

  def stopword_ratio(self, text: str):
    en_stopwords = stopwords.words("english")
    text_split = text.split()
    stop_words = []

    for word in text_split:
      if word in en_stopwords:
        stop_words.append(word)

    word_len = len(text_split)
    stop_ratio = len(stop_words) / word_len
    return stop_ratio

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    df = X.copy()
    df["word_length"] = df["text"].apply(self.word_length)
    df["avg_sentence_length"] = df["text"].apply(self.avg_sentence_length)
    df["punct_ratio"] = df["text"].apply(self.punctuation_ratio)
    df["stopword_ratio"] = df["text"].apply(self.stopword_ratio)
    return df
  