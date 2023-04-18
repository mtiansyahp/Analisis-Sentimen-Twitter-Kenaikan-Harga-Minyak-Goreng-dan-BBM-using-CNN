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
from Modeling import Modeling

from keras.models import load_model


# 

class Main:
    app = Flask(__name__)
    CORS(app)

    global path_model, path_data, path_stopword_list, path_tokenizer
    
    path_model = "../asset/cnn-best.h5"
    path_data = "../asset/datacleanedstemming.csv"
    path_stopword_list = "../asset/idn_stopwords.txt"
    path_tokenizer = "../asset/tokenizer.pickle"

    dataexample = "dasar pemerintah goblok banget dech buat kebijakan aja ga becus!!😘 "
    #Preprocessing Phase ->>

    # data = Preprocessing.clean_tweet(dataexample)



    # load = DataReader.load_stopword_list(path_stopword_list)
    # print(data)
    # 
    




    @app.route('/home', methods=["GET","POST"])
    def predictSentiment():

        ulasan = request.form.get("review")
        print(ulasan)
        model = load_model(path_model, custom_objects={"f1": Modeling.f1, "precision": Modeling.precision, "recall": Modeling.recall})
        tokenizer = DataReader.get_tokenizer(path_tokenizer)
        ulasan_sequence = Modeling.get_seq(ulasan,tokenizer) 
        sentiment_ouput = Modeling.predict_sentiment(model, ulasan_sequence)
        print(sentiment_ouput)
    
    if __name__ == '__main__':
        app.run(debug=True)