import glob
import pprint
import re
import string
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sp
import unicodedata

from scrm.models import DatasetTreito
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stemmer = PorterStemmer()

stopwords_PT = [word.strip() for word in open("utils/stopwords_PT_R.txt").readlines()]
# stopwords_PT = set(stopwords.words('portuguese'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocessing_PT(base):
    # REMOVE URLS
    base['Message'] = base['Message'].replace(to_replace='http\\S+\\s*', value='', regex=True)

    # REMOVE hashtags
    base['Message'] = base['Message'].replace(to_replace='#\\S+', value='', regex=True)

    # REMOVE @mentions
    base['Message'] = base['Message'].replace(to_replace='@\\S+', value='', regex=True)

    # REPLACE ALL PUNCTUATION BY WHITESPACE
    base['Message'] = base['Message'].replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',
                                              regex=True)

    # To Lowercase
    base['Message'] = base['Message'].apply(lambda x: x.lower())

    # REMOVE Stopwords
    base['Message'] = base['Message'].apply(
        lambda x: " ".join([word for word in [item for item in x.split() if item not in stopwords_PT]]))

    # Stemming
    # print(tweets['Message'][22])
    # RSLP Stemming
    stemmer = nltk.stem.RSLPStemmer()

    # Snowball Stemming
    # stemmer = nltk.stem.SnowballStemmer("portuguese")

    base['Message'] = base['Message'].apply(
        lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split() if item not in stopwords_PT]]))

    # REPLACE Portuguese accented characters in R with non-accented counterpart
    base['Message'] = base['Message'].apply(
        lambda x: unicodedata.normalize('NFKD', np.unicode(x)).encode('ASCII', 'ignore'))

    # REMOVE Numbers
    base['Message'] = base['Message'].replace(to_replace='\d+', value='', regex=True)

    return base


def funcTeste(train,test,clf):
    vectorizer = TfidfVectorizer(min_df=1)
    counter = CountVectorizer(min_df=1)

    x_train=vectorizer.fit_transform(train['Message'])
    num_sample,num_features=x_train.shape


    x_test=vectorizer.transform(test)
    test_sample,test_features=x_test.shape

    clf.fit(x_train,train['Truth'])

    prediction = clf.predict(x_test)

    return prediction


def get_multc_fit(clf, dataset):
    vectorizer = TfidfVectorizer(min_df=1)
    x_train = vectorizer.fit_transform(dataset['Message'])
    clf.fit(x_train, dataset['Truth'])
    print("get_multc_fit...OK")
    return clf


def get_svm_predict(clf,test):
    vectorizer = TfidfVectorizer(min_df=1)
    x_test = vectorizer.transform(test)
    predict = clf.predict(x_test)

    return predict


def multiclass(train_dataset_name, label, target,polaridades,processamento):
    train_dataset = DatasetTreito.objects.get(nome__contains=train_dataset_name)
    dataset = pd.read_csv(train_dataset.arquivo)
    dataset = preprocessing_PT(dataset)
    data_01 = target
    target = pd.DataFrame(target, columns=['Message'])
    target = preprocessing_PT(target)
    data = target['Message']
    pprint.pprint(data)

    clf = svm.SVC(kernel='linear')

    predicted = funcTeste(dataset,data,clf)
    for avaliacao,text,polaridade in zip(polaridades,data_01,predicted):
        pprint.pprint(text)
        avaliacao.frase = text
        avaliacao.polaridade = polaridade
        avaliacao.save()
    processamento.concluido=True
    processamento.save()
