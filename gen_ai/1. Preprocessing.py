import pandas as pd
import numpy as np
import re
import os
# import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from json import JSONDecodeError
from httpx import TimeoutException

# Translator: deepl

# import deepl
# auth_key = "f63c02c5-f056-..."
# translator = deepl.Translator(auth_key, verify=False)




# Translator: googletrans 4.0.0

# pip install googletrans-py
#     # need to change line 37 of Translator (client.py): add param verify=False in httpx.Client
from googletrans import Translator
translator = Translator()




# Translator: deep_translator - GoogleTranslator

# from deep_translator import GoogleTranslator
#     # need to change line 68 in GoogleTranslator (google.py): add param verify=False in requests.get

# translator = GoogleTranslator(source='auto', target='en')


base_path = os.path.join(os.getcwd(), r'EY\DATS Germany - TS-SA - TS-SA\03. Data')
    # keep the path of spyder at user level. run the script by selecting all and f9, otherwise the path will change

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def generate_stop():
    words_needed = ['no', 'nor', 'not','only','too', 'very', "don't", "should've", 
                    "aren't", "couldn't", "didn't", "doesn't", "hadn't",
                    "hasn't", "haven't", "isn't", "shouldn't", "wasn't", "weren't", 
                    "won't", "wouldn't"]
    stop_words = [w for w in stopwords.words('english') if w not in words_needed]
    
    punct = [p for p in string.punctuation if p not in '.,!?']
    month = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    day = ['monday', ' tuesday', ' wednesday', ' thursday', ' friday']
    to_be_removed = set(stop_words + punct + month + day)
    return to_be_removed

stop_words = generate_stop()


def translate(comment):
    comment_en = None
    while type(comment_en) is not str:
        try:
            comment_en = translator.translate(comment, src='de').text
        except (JSONDecodeError, TypeError, TimeoutException):
            pass
        # except (ReadTimeout, TimeoutError):
        #     print('.')
        #     return ''
    print('-')
    return comment_en


def clean(comment_en):
    comment_en = re.sub(r'\d+', '', comment_en)
    tokens = [lemmatizer.lemmatize(t) for t in comment_en.lower().split() if t not in stop_words]
    print('-')
    return ' '.join(tokens)


df = pd.read_excel(os.path.join(base_path, 'reviews.xlsx'))
df['Text_en'] = df['Text'].map(translate)
df['Text_clean'] = df['Text_en'].map(clean)
df.to_excel(os.path.join(base_path, 'reviews_cleaned_.xlsx'), index=False)