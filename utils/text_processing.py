import nltk
import nltk.data
import re
#import yandex_translate
#from langdetect import detect
from nltk import wordpunct_tokenize
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import RSLPStemmer as stemmer
from nltk.corpus import stopwords
from utils.read_data import ReadData
from pymongo import MongoClient
import pprint
from utils.db_connection import MongodbConnector


@staticmethod
def language_detector(dict_list):
    updated_dict_list = []
    for dict in dict_list:
        text = dict['text']
        languages_ratios = {}
        tokens = wordpunct_tokenize(text)
        text = re.sub("(?P<url>https?://[^\s]+)", '', text, flags=re.MULTILINE)
        # yandex = yandex_translate.YandexTranslate('trnsl.1.1.20170206T145228Z.a97271f9c4de10c0.a95c985a88ec37045c0069a2e99727434c552bc7')
        words = [word.lower() for word in tokens]

        for language in stopwords.fileids():
            stops = []
            for item in stopwords.words(language):
                stops.extend(wordpunct_tokenize(item))
            stopwords_set = set(stops)
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)

            languages_ratios[language] = len(common_elements)
            most_rated_language = max(languages_ratios, key=languages_ratios.get)
        # text = u''+text
        # emoji_pattern = re.compile("["
        #                            u"\U0001F600-\U0001F64F"  # emoticons
        #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
        #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        #                            "]+", flags=re.UNICODE)
        # text = emoji_pattern.sub(r'', text)
        if text != "":
            if most_rated_language == "portuguese":
                dict['lang'] = "pt"
                updated_dict_list.append(dict)

    return updated_dict_list

@staticmethod
def textNormalizer(dict_list):
    updated_dict_list = []
    for dict in dict_list:
        text = dict['text']
         #hashtags and profile mentions treatment
        dict['hashtags'] = re.findall("\B#\w*[a-zA-Z]+\w*",text)
        #\B#\w*[a-zA-Z]+\w*
        dict['mentions'] = re.findall("\B@\w*[a-zA-Z]+\w*",text)
        for hashtag in dict['hashtags']:
            text = text.replace(hashtag,hashtag.replace('#',''))

        #Slang and Internet common abbreviations treatment

        #word captalization corretion
        #punctuation corretion
        #contraction expansio
        dict['normalized_text'] = text
        updated_dict_list.append(dict)
    return updated_dict_list

@staticmethod
def tokenizer(dict_list):
    pt_tokenizer = nltk.data.load('tokeni')
    updated_dict_list = []
    for dict in dict_list:
        text = dict['normalized_text']
        tokens_list = nltk.word_tokenize(text)
        dict['tokens'] = tokens_list
        updated_dict_list.append(dict)
    return updated_dict_list

def featuresExtrator(dict_list):
    updated_dict_list = []
    for dict in dict_list:
        text = dict['text']
        #POS Tagger
        #Chunken
        # --------------------------------
        #Lemmatization
        wordnet_lemmatizer = WordNetLemmatizer()
        text = text.apply(
            lambda x: " ".join([word for word in [wordnet_lemmatizer.lemmatize(item) for item in x.split()]])
        )
        # --------------------------------
        # --------------------------------
        #Stemming

        text = text.apply(
            lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split()]])
        )
        # ---------------------------------
        #Word shape
        #N-Gram generator
        dict['f_extracted_text'] = text
        updated_dict_list.append(dict)
    return updated_dict_list


#def contextualizer(self):
    #Categories from a Wikipedia concept found in the text
    #Query results from semantic dictionaries

#def dtr(self):



if __name__ == '__main__':
    client = MongoClient()
    db = MongodbConnector.connect_db('socialNet')
    dl = ReadData.twitterDataAll(db)
    # dl_lang = tp.language_detector(dl)
    dl_normalizer = textNormalizer(dl)
    dl_token = tokenizer(dl_normalizer)
    for dict in dl_normalizer:
        pprint.pprint(dict)
