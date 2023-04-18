from typing import Union
from fastapi import FastAPI
# 
import pandas as pd
import numpy as np
import io, nltk, string, re, swifter, gensim
import tensorflow as tf
# 
from flask_cors import CORS
from flask import Flask, request, jsonify, make_response
from flask_restful import reqparse, Api, Resource

from DataReader import DataReader
from Preprocessing import Preprocessing

from gensim.models import word2vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from more_itertools import take #to take particular items from dict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter #to count occurences 

from keras_preprocessing.sequence import pad_sequences  #for padding our text
from keras.preprocessing.text import Tokenizer #for word tokenization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
# 


path_data = "../asset/datacleanedstemming.csv"
path_stopword_list = "../asset/idn_stopwords.txt"

dataexample = "dasar pemerintah goblok banget dech buat kebijakan aja ga becus!!😘 "
#Preprocessing Phase ->>

data = Preprocessing.clean_tweet(dataexample)



# load = DataReader.load_stopword_list(path_stopword_list)
print(data)







app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

