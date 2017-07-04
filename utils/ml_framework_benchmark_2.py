#!/usr/bin/env python
# -*- coding: utf-8 -*-
from importlib import reload

import scipy.sparse as sp
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re
import string
import unicodedata
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import OrderedDict
from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import bigrams

import glob

import pprint

pp = pprint.PrettyPrinter(indent=4)

reload(sys)
sys.setdefaultencoding('utf8')

stopwords_EN = set(stopwords.words('english'))

stemmer = PorterStemmer()

wordnet_lemmatizer = WordNetLemmatizer()


# print "It works!"
# sys.exit()

def preprocessing_EN(tweets, tolower, remove_url, remove_hashtag, remove_mentions, remove_punc, remove_stopwords,
                     stemming, lemmatize, norm_accent, remove_numbers, language):
    # OBS: This is only for testing benchmark, to remove new lines and line breaks from Cornel Movie reviews
    # tweets['Message'] = tweets['Message'].replace(to_replace='\n', value=' ',regex=True)
    # tweets['Message'] = tweets['Message'].replace(to_replace='\r', value=' ',regex=True)

    # To Lowercase
    if tolower:
        tweets['Message'] = tweets['Message'].apply(lambda x: x.lower())

    # REMOVE URLS
    if remove_url:
        tweets['Message'] = tweets['Message'].replace(to_replace='http\\S+\\s*', value='', regex=True)

    # REMOVE hashtags
    if remove_hashtag:
        tweets['Message'] = tweets['Message'].replace(to_replace='#\\S+', value='', regex=True)

    # REMOVE @mentions
    if remove_mentions:
        tweets['Message'] = tweets['Message'].replace(to_replace='@\\S+', value='', regex=True)

    # REPLACE ALL PUNCTUATION BY WHITESPACE
    if remove_punc:
        tweets['Message'] = tweets['Message'].replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',
                                                      regex=True)

    # REMOVE Stopwords
    if remove_stopwords:
        tweets['Message'] = tweets['Message'].apply(
            lambda x: " ".join([word for word in [item for item in x.split() if item not in stopwords_EN]]))

    # Stemming
    if stemming:
        # print(tweets['Message'][22])
        # RSLP Stemming
        # stemmer = nltk.stem.RSLPStemmer()

        # Snowball Stemming
        # stemmer = nltk.stem.SnowballStemmer("portuguese")

        tweets['Message'] = tweets['Message'].apply(
            lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split()]]))

    # Lemmatizer
    if lemmatize:
        tweets['Message'] = tweets['Message'].apply(
            lambda x: " ".join([word for word in [wordnet_lemmatizer.lemmatize(item) for item in x.split()]]))

    # REPLACE Portuguese accented characters in R with non-accented counterpart
    if norm_accent:
        tweets['Message'] = tweets['Message'].apply(
            lambda x: unicodedata.normalize('NFKD', np.unicode(x)).encode('ASCII', 'ignore'))

    # REMOVE Numbers
    if remove_numbers:
        tweets['Message'] = tweets['Message'].replace(to_replace='\d+', value='', regex=True)

    return tweets


def vectorizeTFIDF(data, ngram_range):
    if ngram_range == (1, 2, 3):
        ngram_range = (1, 3)

        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        X = vectorizer.fit_transform(data)

        # TFIDF weights
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

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


def benchmark():
    """ #################### Input Data and Initial Settings ########################### """

    datasets_dir = "datasets/datasets_csv_90"

    datasets = glob.glob(datasets_dir + "/*.csv")

    """ #################### PRE-PROCESSING ########################### """
    """
    tolower = True
    remove_url = True
    remove_hashtag = True
    remove_mentions = True
    remove_punc = True
    remove_stopwords = False
    stemming = True
    lemmatize = False
    norm_accent = True
    remove_numbers = True
    language = "en"
    """

    """ #################### DATA TO OUTPUT ########################### """
    dataset_vec = []
    scenario_vec = []
    classification_label = []
    methods_vec = []
    accuracy_vec = []
    precision_pos_vec = []
    precision_neg_vec = []
    recall_pos_vec = []
    recall_neg_vec = []
    f1_score_pos_vec = []
    f1_score_neg_vec = []
    macro_f1_vec = []

    dict_methods_results = defaultdict(lambda x: [])

    """ #################### FEATURE EXTRACTION ########################### """
    tf = False
    tf_idf = True

    """ Benchmark Scenarios """
    scenarios = [
        {"scenario_name": "s1", "tolower": True, "remove_url": True, "remove_hashtag": True, "remove_mentions": True,
         "remove_punc": True, "remove_stopwords": False, "stemming": True,
         "lemmatize": False, "norm_accent": True, "remove_numbers": True, "language": "True"},

        {"scenario_name": "s2", "tolower": True, "remove_url": True, "remove_hashtag": True, "remove_mentions": True,
         "remove_punc": True, "remove_stopwords": False, "stemming": False,
         "lemmatize": True, "norm_accent": True, "remove_numbers": True, "language": "True"},

        {"scenario_name": "s3", "tolower": True, "remove_url": True, "remove_hashtag": True, "remove_mentions": True,
         "remove_punc": True, "remove_stopwords": True, "stemming": True,
         "lemmatize": False, "norm_accent": True, "remove_numbers": True, "language": "True"},

        {"scenario_name": "s4", "tolower": True, "remove_url": True, "remove_hashtag": True, "remove_mentions": True,
         "remove_punc": True, "remove_stopwords": True, "stemming": False,
         "lemmatize": True, "norm_accent": True, "remove_numbers": True, "language": "True"}
        ]

    nclasses = 2

    labels_order_2 = [1, -1]

    r_vec = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (1, 2, 3)]
    fname_vec = ["UNIGRAM", "BIGRAM", "TRIGRAM", "UNI_BI", "UNI_TRI", "UNI_BI_TRI"]

    classification_methods = ["multiclass", "oneclass_svm"]

    for dataset in datasets:

        print("Dataset: %s" % dataset)

        tweets = pd.read_csv(dataset)

        if len(set(tweets["Truth"])) > 2:
            continue

        train_idx_pos = tweets[(tweets['Experiment'] == "train_pos")].index.tolist()
        test_idx_pos = tweets[(tweets['Experiment'] == "test_pos")].index.tolist()

        train_idx_neg = tweets[(tweets['Experiment'] == "train_neg")].index.tolist()
        test_idx_neg = tweets[(tweets['Experiment'] == "test_neg")].index.tolist()

        # Indexes for Multiclass
        train_idx = train_idx_pos + train_idx_neg
        test_idx = test_idx_pos + test_idx_neg

        X_pos_outliers = test_idx_neg
        X_neg_outliers = test_idx_pos

        # Save predictions for ensemble approach
        all_predictions_dict = {}

        # Creat y (labels)
        y = tweets['Truth']

        # Set data for all scenarios beforehand
        # X_vec = []
        X_vec = defaultdict(lambda: [])

        for scenario in scenarios:

            print("Scenario: %s" % scenario["scenario_name"])

            # Preprocessing
            tweets = preprocessing_EN(tweets, scenario["tolower"], scenario["remove_url"], scenario["remove_hashtag"],
                                      scenario["remove_mentions"]
                                      , scenario["remove_punc"], scenario["remove_stopwords"], scenario["stemming"],
                                      scenario["lemmatize"], scenario["norm_accent"], scenario["remove_numbers"],
                                      scenario["language"])

            """ #################### Feature Vector ########################### """
            # Feature Extraction
            # Create beforehand the X matrices with different feature extraction settings and save them
            for r, fname in zip(r_vec, fname_vec):
                X = vectorizeTFIDF(tweets['Message'], r)
                X_vec[scenario["scenario_name"]].append(X)

        print("Vectorizing done for all scenarios.")

        for cm in classification_methods:

            if cm == "multiclass":

                """ #################### Multiclass Classifiers ########################### """

                clf1 = svm.SVC(kernel='linear')
                clf2 = MultinomialNB()
                clf3 = tree.DecisionTreeClassifier()
                clf4 = LogisticRegression(random_state=1)
                clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
                """ Ensemble Classifier """
                eclf = VotingClassifier(
                    estimators=[('svm', clf1), ('mnb', clf2), ('dt', clf3), ('gnb', clf4), ('nnet', clf5)],
                    voting='hard')

                dict_methods = {"SVM": svm.SVC(kernel='linear'),
                                "MNB": MultinomialNB(),
                                "DT": tree.DecisionTreeClassifier(),
                                "LR": LogisticRegression(random_state=1),
                                "NNET": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),
                                                      random_state=1),
                                "Ensemble": VotingClassifier(
                                    estimators=[('svm', clf1), ('mnb', clf2), ('dt', clf3), ('gnb', clf4),
                                                ('nnet', clf5)], voting='hard')}

                for scenario in scenarios:

                    for label in ['SVM', 'MNB', 'DT', 'LR', 'NNET', 'Ensemble']:

                        for X, fname in zip(X_vec[scenario["scenario_name"]], fname_vec):
                            clf = dict_methods[label]

                            clf.fit(X[train_idx], y[train_idx])
                            predicted = clf.predict(X[test_idx])

                            accuracy = accuracy_score(y[test_idx], predicted)
                            precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted,
                                                                                                 labels=labels_order_2)
                            macro_f1 = f1_score(y[test_idx], predicted, average='macro')

                            """
                            accuracy = 1.
                            precision, recall, fscore, support = [(1, 1), (1, 1), (1, 1), 1]
                            macro_f1 = 1.
                            """

                            dataset_vec.append(dataset)
                            classification_label.append(cm)
                            scenario_vec.append(scenario["scenario_name"])

                            method_name = "_".join([label, fname])
                            methods_vec.append(method_name)

                            accuracy_vec.append(accuracy)
                            precision_pos_vec.append(precision[0])
                            precision_neg_vec.append(precision[1])
                            recall_pos_vec.append(recall[0])
                            recall_neg_vec.append(recall[1])
                            f1_score_pos_vec.append(fscore[0])
                            f1_score_neg_vec.append(fscore[1])
                            macro_f1_vec.append(macro_f1)

                        print("Method %s done." % label)

                    print("Scenario %s Multiclass classification done." % scenario["scenario_name"])

                print("Multiclass classification done.")


            elif cm == "oneclass_svm":

                for scenario in scenarios:

                    # fit the model for POSITIVES
                    label = "OCC_SVM_POS"

                    for X, fname in zip(X_vec[scenario["scenario_name"]], fname_vec):
                        clf_pos = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                        clf_pos.fit(X[train_idx_pos,], y[train_idx_pos])

                        y_pred_train_pos = clf_pos.predict(X[train_idx_pos,])
                        y_pred_test_pos = clf_pos.predict(X[test_idx_pos,])
                        y_pred_outliers_pos = clf_pos.predict(X[test_idx_neg,])

                        y_all_preds_pos = np.concatenate([y_pred_test_pos, y_pred_outliers_pos])
                        y_all_test_idx_pos = test_idx_pos + test_idx_neg

                        accuracy_all_svm_pos = accuracy_score(y[y_all_test_idx_pos], y_all_preds_pos)
                        precision_svm_pos, recall_svm_pos, fscore_svm_pos, support_svm_pos = precision_recall_fscore_support(
                            y[y_all_test_idx_pos], y_all_preds_pos, labels=labels_order_2)
                        macro_f1_svm_pos = f1_score(y[y_all_test_idx_pos], y_all_preds_pos, average='macro')

                        """
                        accuracy_all_svm_pos = 1.
                        precision_svm_pos, recall_svm_pos, fscore_svm_pos, support_svm_pos = [(1,1), (1,1), (1,1), 1.]
                        macro_f1_svm_pos = 1.
                        """

                        dataset_vec.append(dataset)
                        classification_label.append(cm)
                        scenario_vec.append(scenario["scenario_name"])

                        method_name = "_".join([label, fname])
                        methods_vec.append(method_name)

                        accuracy_vec.append(accuracy_all_svm_pos)
                        precision_pos_vec.append(precision_svm_pos[0])
                        precision_neg_vec.append(precision_svm_pos[1])
                        recall_pos_vec.append(recall_svm_pos[0])
                        recall_neg_vec.append(recall_svm_pos[1])
                        f1_score_pos_vec.append(fscore_svm_pos[0])
                        f1_score_neg_vec.append(fscore_svm_pos[1])
                        macro_f1_vec.append(macro_f1_svm_pos)

                    # fit the model for NEGATIVES
                    label = "OCC_SVM_NEG"

                    for X, fname in zip(X_vec[scenario["scenario_name"]], fname_vec):
                        clf_neg = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                        clf_neg.fit(X[train_idx_neg,])

                        y_pred_train_neg = clf_neg.predict(X[train_idx_neg,])
                        y_pred_test_neg = clf_neg.predict(X[test_idx_neg,])
                        y_pred_outliers_neg = clf_neg.predict(X[test_idx_pos,])

                        y_all_preds_neg = np.concatenate([y_pred_test_neg, y_pred_outliers_neg])
                        y_all_test_idx_neg = test_idx_neg + test_idx_pos

                        accuracy_all_svm_neg = accuracy_score(y[y_all_test_idx_neg], y_all_preds_neg)
                        precision_svm_neg, recall_svm_neg, fscore_svm_neg, support_svm_neg = precision_recall_fscore_support(
                            y[y_all_test_idx_neg], y_all_preds_neg, labels=labels_order_2)
                        macro_f1_svm_neg = f1_score(y[y_all_test_idx_neg], y_all_preds_neg, average='macro')

                        dataset_vec.append(dataset)
                        classification_label.append(cm)
                        scenario_vec.append(scenario["scenario_name"])

                        method_name = "_".join([label, fname])
                        methods_vec.append(method_name)

                        accuracy_vec.append(accuracy_all_svm_neg)
                        precision_pos_vec.append(precision_svm_neg[0])
                        precision_neg_vec.append(precision_svm_neg[1])
                        recall_pos_vec.append(recall_svm_neg[0])
                        recall_neg_vec.append(recall_svm_neg[1])
                        f1_score_pos_vec.append(fscore_svm_neg[0])
                        f1_score_neg_vec.append(fscore_svm_neg[1])
                        macro_f1_vec.append(macro_f1_svm_neg)

                    print("Scenario %s OneClass classification done." % scenario["scenario_name"])

                print("OneClass classification done.")


            elif cm == "oneclass_positivenaivebayes":
                pass

            elif cm == "ensemble":
                # all_truth = y[test_idx]

                """
                for test_instance in all_truth:

                    # you have to save all predictions for all methods, parallelly 
                    # Count final labels for each test instance
                    # Give final label based on majority
                    # Use final label vector to compute all metrics with all_truth vector

                """

                pass

            else:
                print("Invalid classification method")
                sys.exit()

    """ #################### Export Results ########################### """

    results_df_detailed = OrderedDict([
        ('dataset', dataset_vec),
        ( 'classification', classification_label),
        ( 'scenario', scenario_vec ),
        ('method', methods_vec),
        ('accuracy', accuracy_vec),
        ('prec_pos', precision_pos_vec),
        ( 'rec_pos', recall_pos_vec),
        ('f1_pos', f1_score_pos_vec),
        ('prec_neg', precision_neg_vec),
        ('rec_neg', recall_neg_vec),
        ('f1_neg', f1_score_neg_vec),
        ('macro_f1', macro_f1_vec)
    ])

    df = pd.DataFrame(results_df_detailed)

    df.to_csv( \
    analytics_results_filename, sep=',')


if __name__ == '__main__':

    analytics_results_filename = "benchmark_results/benchmark_results_5.csv"

    benchmark()