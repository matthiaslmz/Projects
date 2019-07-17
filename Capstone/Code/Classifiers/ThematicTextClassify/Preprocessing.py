
# general imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# gensim imports for preprocessing
import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# Using POS Tag to lemmatize
from nltk.corpus import wordnet

# get pos tag for lemmetization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# remove non ascii characters
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

stop_words = gensim.parsing.preprocessing.STOPWORDS 
# We considred using a stemmer, but since we have the pos tags 
# creating a more accuracte lemmatizer we will not go down that process
# better and newwer stemmer than porterstemmer 
stemmer = SnowballStemmer('english')
# stemmer only used for example and comparison between stemmning and lemmatizing in notebook

# wrap everythin in a preprocessing_text function
def preprocess_text(text):
    processed = []
    for token in gensim.utils.simple_preprocess(text):
        # we dont want words that have length less than 3 since it usually doesn't make sense
        if token not in stop_words  and len(token) > 3 and is_ascii(token) == True:
            processed.append(WordNetLemmatizer().lemmatize(token, pos = get_wordnet_pos(token)))
    return processed

