from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from DataReader import DataReader
import re, string, swifter
import pandas as pd


class Preprocessing:
    def remove_duplicates (df, df_subset):
        df.drop_duplicates(subset= df_subset, keep='last',inplace=True)
        df.reset_index(drop=True)
    
        return df

    # def clean_tweet(teks):

    #     '''Make tweet lowercase, remove punctuation and remove words containing numbers. remove newline'''

    #     teks = str(teks)
    #     teks = teks.encode('utf-8').decode('ascii', 'ignore')
    #     teks = teks.lower()
    #     teks = re.sub('[%s]' % re.escape(string.punctuation.replace('?', '')), '', teks)
    #     text = re.sub("[^a-zA-Z0-9\s:\n\\n]+@]", '', teks) 
    #     teks = re.sub('http\S+', '', teks) # remove links
    #     teks = re.sub('www\S+', '', teks)
    #     teks = re.sub('\w*\d\w*', '', teks)
    #     teks = re.sub('[‘’“”…]', '', teks)
    #     teks = re.sub('\n', '', teks)
    #     teks = re.sub('\r', '', teks)
    #     teks = teks.replace('?', ' ?')
    #     teks = teks.replace('\d+', '')
    #     teks = re.sub('[.;:!\'?,\"()\[\]*~]', '', teks)
    #     teks = re.sub('(<br\s*/><br\s*/>)|(\-)|(\/)', '', teks)
    #     teks = re.sub('\n2', '', teks)
    
    #     return teks

    
    def stopword_removal(text):
        word = [word for word in text.split() if word not in idn_stopwords]
        result = ' '.join(word)
        
        return result


    def stemming (text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        word = [stemmer.stem(word) for word in text.split()]
        result = ' '.join(word)

        return result

    
    

    
