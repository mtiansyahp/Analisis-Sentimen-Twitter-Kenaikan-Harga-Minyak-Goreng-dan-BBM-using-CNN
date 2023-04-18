import pandas as pd
import pickle 

# 
class DataReader:

    def get_data(path):
        data = pd.read_csv(path)
        return data

    def load_slang_word(path):
        slang_words = {}
        data = pd.read_csv(path)
        for i, row in data.iterrows():
            slang_words[row["slang"]] = row["formal"]
        return data
    
    #function to replace certain value in dictionary with other value
    def replace_value_in_dict (to_change, new_value, dict):
        for value in dict.values():
            if value == to_change:
                target_key = list(dict.keys())[list(dict.values()).index(to_change)] #get all dict keys that has 'enggak' value (it could be any value that we want to change)
                dict[target_key] = new_value #assign the new value
                
        return dict
    
    def replace_slang_words (text):
        word = [word.replace(word, slang_words[word]) if word in slang_words else word for word in text.split() ]
        result = ' '.join(word)

        return result
    
    def get_tokenizer(path):
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        return tokenizer

    def load_stopword_list(path):
        stopword_file = open(path, "r")
        # reading the file
        stopword_list = stopword_file.read()
        # replacing end splitting the text when newline ('\n') is seen.
        final_stopword_list = stopword_list.split("\n")

        return final_stopword_list

    





