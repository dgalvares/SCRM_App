import glob
import pprint
import re
import string

import numpy as np
import pandas as pd
import scipy.sparse as sp
import unicodedata

from scrm.models import DatasetTreito
from collections import defaultdict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stemmer = PorterStemmer()
stopwords_EN = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocessing_EN(base):

    base = base.replace(to_replace='http\\S+\\s*', value='', regex=True)

    base = base.replace(to_replace='#\\S+', value='', regex=True)

    base = base.replace(to_replace='@\\S+', value='', regex=True)

    base = base.replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',
                                                      regex=True)

    base = base.replace(to_replace='\d+', value='', regex=True)
    print("preprocessing...OK")
    return base


def vectorizeTFIDF_train(data, ngram_range):
    if ngram_range == (1, 2, 3):
        ngram_range = (1, 3)

        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        # vectorizer = TfidfVectorizer(min_df=1,ngram_range=ngram_range)

        X = vectorizer.fit_transform(data)
        # X = vectorizer.transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)
        # X = transformer.transform(X)

        return X

    if ngram_range == (1, 3):

        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
        X = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

        ngram_range = (3, 3)

        vectorizer_3 = CountVectorizer(min_df=1, ngram_range=(3, 3))
        X_3 = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer_3 = TfidfTransformer()
        X_3 = transformer.fit_transform(X_3)

        X_final = sp.hstack((X, X_3), format='csr')

        X = X_final

    else:
        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        X = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

    return X


def vectorizeTFIDF_test(data, ngram_range):
    if ngram_range == (1, 2, 3):
        ngram_range = (1, 3)

        # vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        vectorizer = TfidfVectorizer(min_df=1,ngram_range=ngram_range)

        # X = vectorizer.fit_transform(data)
        X = vectorizer.transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        # X = transformer.fit_transform(X)
        X = transformer.transform(X)

        return X

    if ngram_range == (1, 3):

        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
        X = vectorizer.fit_transform(data)


        # TFIDF weights
        transformer = TfidfTransformer()
        # X = transformer.fit_transform(X)
        X = transformer.transform(X)


        ngram_range = (3, 3)

        vectorizer_3 = CountVectorizer(min_df=1, ngram_range=(3, 3))
        X_3 = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer_3 = TfidfTransformer()
        X_3 = transformer.fit_transform(X_3)

        X_final = sp.hstack((X, X_3), format='csr')

        X = X_final

    else:
        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        X = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

    return X



def get_multc_fit(clf, dataset):

    y = dataset['Truth']
    X = to_vector(dataset['Message'], 1)
    clf.fit(X, y)
    print("get_multc_fit...OK")
    return clf

def to_vector(dataset,opt):
    r_vec = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (1, 2, 3)]
    fname_vec = ["UNIGRAM", "BIGRAM", "TRIGRAM", "UNI_BI", "UNI_TRI", "UNI_BI_TRI"]

    for r, fname in zip(r_vec, fname_vec):
        if(opt ==1):
            X = vectorizeTFIDF_train(dataset, r)
        else:
            X = vectorizeTFIDF_test(dataset, r)
    return X


def multiclass(train_dataset_name, label, target):
    train_dataset = DatasetTreito.objects.get(nome__contains=train_dataset_name)
    dataset = pd.read_csv(train_dataset.arquivo)
    clf = get_multc_fit(svm.SVC(kernel='linear'),dataset)
    # target = pd.DataFrame(target)
    # target = preprocessing_EN(target)
    target = to_vector(target, 2)
    predicted = clf.predict(target)
    pprint.pprint(predicted)
    return predicted
