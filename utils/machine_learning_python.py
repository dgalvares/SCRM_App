#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import matplotlib.pyplot as plt

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

# TODO

# 1 - One class classifier - Pos ou Neg
# 2 - Performance pra cada um da classe
# 3 - Performance ensemble
# 4 - 


# 5 - 



# 1 - Hipoteses das ideias

# 2 - Se tem trabalhos correlatos

# 3 - Procurar diferencial 



# AMANHA

# 1 - Write research goals from everything Fabio said

# LATER

# 2 - Later, look if there are papers working with those RG

# 3 - The work will be with a dataset validated in the literature



reload(sys)  
sys.setdefaultencoding('utf8')


""" #################### Input Data and Initial Settings ########################### """

# This makes sure the experiment`ll be always reproducible
#np.random.seed(2017)

input_pos = "twitter_data_final/twitter_data_POSITIVE_REVISED.csv"
input_neg = "twitter_data_final/twitter_data_NEGATIVE_REVISED.csv"
input_neu = "twitter_data_final/twitter_data_NEUTRAL_REVISED.csv"

input_final = "twitter_data_final/vader_twitter_CSV_FINAL.csv"

stopwords_PT = [word.strip() for word in open("stopwords_PT_R.txt").readlines()]
stopwords_EN = set(stopwords.words('english'))


tweets = pd.read_csv(input_final)
analytics_results_file_name = "final_code_results/2classes/results_2classes_diffFoldForEach_10Fold_VADER_TWITTER.csv"

""" #################### PRE-PROCESSING ########################### """

"""
# REMOVE URLS
#gsub('http\\S+\\s*', '', sentence)
tweets_all$Message <- gsub('http\\S+\\s*', '', tweets_all$Message)

# REMOVE hashtags
tweets_all$Message <- gsub('#\\S+', '', tweets_all$Message)

# REMOVE @mentions
tweets_all$Message <- gsub('@\\S+', '', tweets_all$Message)

# REPLACE ALL PUNCTUATION BY WHITESPACE
tweets_all$Message <- gsub("[[:punct:]]", " ", tweets_all$Message)

# REPLACE Portuguese accented characters in R with non-accented counterpart
tweets_all$Message <- iconv(tweets_all$Message, to='ASCII//TRANSLIT')


corpus <- Corpus(VectorSource(tweets_all$Message))
corpus <- tm_map(corpus, content_transformer(tolower))
#corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, iconv(stopwords(kind = "pt"), to='ASCII//TRANSLIT'))
corpus <- tm_map(corpus, stemDocument, language = "portuguese")
corpus <- tm_map(corpus, PlainTextDocument)
#corpus <- tm_map(corpus, removeWords, c(stopwords(kind = "pt"),"boa", "pra", "todos", "bom", "ser", "vai", "ainda", "bem"))
#matrix <- DocumentTermMatrix(corpus, control = list(tokenize = mytokenizer))
matrix <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
"""

def preprocessing_PT(tweets):

    # REMOVE URLS
    tweets['Message'] = tweets['Message'].replace(to_replace='http\\S+\\s*', value='',regex=True)

    # REMOVE hashtags
    tweets['Message'] = tweets['Message'].replace(to_replace='#\\S+', value='',regex=True)


    # REMOVE @mentions
    tweets['Message'] = tweets['Message'].replace(to_replace='@\\S+', value='',regex=True)

    # REPLACE ALL PUNCTUATION BY WHITESPACE
    tweets['Message'] = tweets['Message'].replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',regex=True)

    # To Lowercase
    tweets['Message'] = tweets['Message'].apply(lambda x: x.lower())

    # REMOVE Stopwords
    tweets['Message'] = tweets['Message'].apply(lambda x: " ".join([word for word in [item for item in x.split() if item not in stopwords_PT]]))

    # Stemming
    #print(tweets['Message'][22])
    # RSLP Stemming 
    stemmer = nltk.stem.RSLPStemmer()

    # Snowball Stemming
    #stemmer = nltk.stem.SnowballStemmer("portuguese")

    tweets['Message'] = tweets['Message'].apply(lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split() if item not in stopwords_PT]]))


    # REPLACE Portuguese accented characters in R with non-accented counterpart
    tweets['Message'] = tweets['Message'].apply(lambda x: unicodedata.normalize('NFKD', unicode(x)).encode('ASCII','ignore'))

    # REMOVE Numbers
    tweets['Message'] = tweets['Message'].replace(to_replace='\d+', value='',regex=True)


    return tweets



def preprocessing_EN(tweets):

    # REMOVE URLS
    tweets['Message'] = tweets['Message'].replace(to_replace='http\\S+\\s*', value='',regex=True)

    # REMOVE hashtags
    tweets['Message'] = tweets['Message'].replace(to_replace='#\\S+', value='',regex=True)


    # REMOVE @mentions
    tweets['Message'] = tweets['Message'].replace(to_replace='@\\S+', value='',regex=True)

    # REPLACE ALL PUNCTUATION BY WHITESPACE
    tweets['Message'] = tweets['Message'].replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',regex=True)

    # To Lowercase
    tweets['Message'] = tweets['Message'].apply(lambda x: x.lower())

    # REMOVE Stopwords
    tweets['Message'] = tweets['Message'].apply(lambda x: " ".join([word for word in [item for item in x.split() if item not in stopwords_EN]]))

    # Stemming
    #print(tweets['Message'][22])
    # RSLP Stemming 
    #stemmer = nltk.stem.RSLPStemmer()

    # Snowball Stemming
    #stemmer = nltk.stem.SnowballStemmer("portuguese")

    stemmer = PorterStemmer()

    tweets['Message'] = tweets['Message'].apply(lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split() if item not in stopwords_EN]]))


    # REPLACE Portuguese accented characters in R with non-accented counterpart
    tweets['Message'] = tweets['Message'].apply(lambda x: unicodedata.normalize('NFKD', unicode(x)).encode('ASCII','ignore'))

    # REMOVE Numbers
    tweets['Message'] = tweets['Message'].replace(to_replace='\d+', value='',regex=True)


    return tweets


tweets = preprocessing_EN(tweets)


#print(" ".join(nltk.stem.PorterStemmer.languages))

#print(tweets['Message'][22])

""" #################### Feature Vector ########################### """


vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
X = vectorizer.fit_transform(tweets['Message'])

# TFIDF weights
transformer = TfidfTransformer()
X = transformer.fit_transform(X)

# Creat y (labels)
y = tweets['Truth']


""" #################### Classifiers ########################### """

clf1 = svm.SVC(kernel='linear')
clf2 = MultinomialNB()
clf3 = tree.DecisionTreeClassifier()
clf4 = LogisticRegression(random_state=1)
# TODO Understand Neural Network settings
clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


""" Ensemble Classifier """
# TODO Don`t know if I have to create it now or after running all methods. Try this first
eclf = VotingClassifier(estimators=[('svm', clf1), ('mnb', clf2), ('dt', clf3),('gnb', clf4), ('nnet', clf5)], voting='hard')


""" Compute average cross-validation performance metrics """

nfolds = 10
#cv = KFold(len(y), nfolds)



# 3 Classes
labels_order_3 = [1, -1, 0]
# 2 Classes
labels_order_2 = [1, -1]



# SVM Data
accuracies_svm = []
precisions_pos_svm = []
precisions_neg_svm = []
precisions_neu_svm = []
recalls_pos_svm = []
recalls_neg_svm = []
recalls_neu_svm = []
f1_scores_pos_svm = []
f1_scores_neg_svm = []
f1_scores_neu_svm = []
macro_f1_vec_svm = []


# MNB Data
accuracies_mnb = []
precisions_pos_mnb = []
precisions_neg_mnb = []
precisions_neu_mnb = []
recalls_pos_mnb = []
recalls_neg_mnb = []
recalls_neu_mnb = []
f1_scores_pos_mnb = []
f1_scores_neg_mnb = []
f1_scores_neu_mnb = []
macro_f1_vec_mnb = []


# Decision Tree Data
accuracies_dt = []
precisions_pos_dt = []
precisions_neg_dt = []
precisions_neu_dt = []
recalls_pos_dt = []
recalls_neg_dt = []
recalls_neu_dt = []
f1_scores_pos_dt = []
f1_scores_neg_dt = []
f1_scores_neu_dt = []
macro_f1_vec_dt = []


# Logistic Regression Data
accuracies_lr = []
precisions_pos_lr = []
precisions_neg_lr = []
precisions_neu_lr = []
recalls_pos_lr = []
recalls_neg_lr = []
recalls_neu_lr = []
f1_scores_pos_lr = []
f1_scores_neg_lr = []
f1_scores_neu_lr = []
macro_f1_vec_lr = []


# Neural Network Data
accuracies_nnet = []
precisions_pos_nnet = []
precisions_neg_nnet = []
precisions_neu_nnet = []
recalls_pos_nnet = []
recalls_neg_nnet = []
recalls_neu_nnet = []
f1_scores_pos_nnet = []
f1_scores_neg_nnet = []
f1_scores_neu_nnet = []
macro_f1_vec_nnet = []


# Ensemble Data
accuracies_ens = []
precisions_pos_ens = []
precisions_neg_ens = []
precisions_neu_ens = []
recalls_pos_ens = []
recalls_neg_ens = []
recalls_neu_ens = []
f1_scores_pos_ens = []
f1_scores_neg_ens = []
f1_scores_neu_ens = []
macro_f1_vec_ens = []

""" ****************************************** 3 CLASSES ************************************************ """

""" #################### Classifiers APPROACH 1 - ALL METHODS WITH SAME FOLDS ########################### """
'''
# StratifiedKFold is a variation of k-fold which returns stratified folds: each set 
#contains approximately the same percentage of samples of each target class as the complete set.
skf = StratifiedKFold(n_splits=nfolds)


#for train_idx, test_idx in cv:
for train_idx, test_idx in skf.split(X, y):
   
    # TODO Keep working here, get every score, save in lists, compute the mean, find out how to plot

    # SVM
    clf1.fit(X[train_idx], y[train_idx])
    predicted = clf1.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_svm.append(accuracy)
    precisions_pos_svm.append(precision[0])
    precisions_neg_svm.append(precision[1])
    precisions_neu_svm.append(precision[2])
    recalls_pos_svm.append(recall[0])
    recalls_neg_svm.append(recall[1])
    recalls_neu_svm.append(recall[2])
    f1_scores_pos_svm.append(fscore[0])
    f1_scores_neg_svm.append(fscore[1])
    f1_scores_neu_svm.append(fscore[2])
    macro_f1_vec_svm.append(macro_f1)


    # MNB
    clf2.fit(X[train_idx], y[train_idx])
    predicted = clf2.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_mnb.append(accuracy)
    precisions_pos_mnb.append(precision[0])
    precisions_neg_mnb.append(precision[1])
    precisions_neu_mnb.append(precision[2])
    recalls_pos_mnb.append(recall[0])
    recalls_neg_mnb.append(recall[1])
    recalls_neu_mnb.append(recall[2])
    f1_scores_pos_mnb.append(fscore[0])
    f1_scores_neg_mnb.append(fscore[1])
    f1_scores_neu_mnb.append(fscore[2])
    macro_f1_vec_mnb.append(macro_f1)


    # Decision Tree
    clf3.fit(X[train_idx], y[train_idx])
    predicted = clf3.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_dt.append(accuracy)
    precisions_pos_dt.append(precision[0])
    precisions_neg_dt.append(precision[1])
    precisions_neu_dt.append(precision[2])
    recalls_pos_dt.append(recall[0])
    recalls_neg_dt.append(recall[1])
    recalls_neu_dt.append(recall[2])
    f1_scores_pos_dt.append(fscore[0])
    f1_scores_neg_dt.append(fscore[1])
    f1_scores_neu_dt.append(fscore[2])
    macro_f1_vec_dt.append(macro_f1)


    # Logistic Regression
    clf4.fit(X[train_idx], y[train_idx])
    predicted = clf4.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_lr.append(accuracy)
    precisions_pos_lr.append(precision[0])
    precisions_neg_lr.append(precision[1])
    precisions_neu_lr.append(precision[2])
    recalls_pos_lr.append(recall[0])
    recalls_neg_lr.append(recall[1])
    recalls_neu_lr.append(recall[2])
    f1_scores_pos_lr.append(fscore[0])
    f1_scores_neg_lr.append(fscore[1])
    f1_scores_neu_lr.append(fscore[2])
    macro_f1_vec_lr.append(macro_f1)


    # Neural Network
    clf5.fit(X[train_idx], y[train_idx])
    predicted = clf5.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_nnet.append(accuracy)
    precisions_pos_nnet.append(precision[0])
    precisions_neg_nnet.append(precision[1])
    precisions_neu_nnet.append(precision[2])
    recalls_pos_nnet.append(recall[0])
    recalls_neg_nnet.append(recall[1])
    recalls_neu_nnet.append(recall[2])
    f1_scores_pos_nnet.append(fscore[0])
    f1_scores_neg_nnet.append(fscore[1])
    f1_scores_neu_nnet.append(fscore[2])
    macro_f1_vec_nnet.append(macro_f1)

    
    # Ensemble
    eclf.fit(X[train_idx], y[train_idx])
    predicted = eclf.predict(X[test_idx])
    
    accuracy = accuracy_score(y[test_idx], predicted)
    precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
    macro_f1 = f1_score(y[test_idx], predicted, average='macro')

    accuracies_ens.append(accuracy)
    precisions_pos_ens.append(precision[0])
    precisions_neg_ens.append(precision[1])
    precisions_neu_ens.append(precision[2])
    recalls_pos_ens.append(recall[0])
    recalls_neg_ens.append(recall[1])
    recalls_neu_ens.append(recall[2])
    f1_scores_pos_ens.append(fscore[0])
    f1_scores_neg_ens.append(fscore[1])
    f1_scores_neu_ens.append(fscore[2])
    macro_f1_vec_ens.append(macro_f1)

    #print accuracy
    #print precision
    #print recall
    #print fscore
    #print macro_f1

    #sys.exit()
'''


""" #################### Classifiers APPROACH 2 - ONE LOOP AND DIFF FOLDS FOR EACH METHOD ########################### """
'''
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['SVM', 'MNB', 'DT', 'LR', 'NNET', 'Ensemble']):

	accuracies = []
	precisions_pos = []
	precisions_neg = []
	precisions_neu = []
	recalls_pos = []
	recalls_neg = []
	recalls_neu = []
	f1_scores_pos = []
	f1_scores_neg = []
	f1_scores_neu = []
	macro_f1_vec = []


	# StratifiedKFold is a variation of k-fold which returns stratified folds: each set 
	#contains approximately the same percentage of samples of each target class as the complete set.
	skf = StratifiedKFold(n_splits=nfolds)

	for train_idx, test_idx in skf.split(X, y):



		# TODO Vai dar certo!!! So precisa ver como pegar probabilidade de predicao pra cada classe (y_score = clf.fit(X[train_idx], y[train_idx]).decision_function(X[test_idx]))

		#print y[test_idx]

		#print "=============================================="

		#print y[test_idx][y == 0]



		#sys.exit()

		

		#clf_temp = clf

		#y_score = clf_temp.fit(X[train_idx], y[train_idx]).decision_function(X[test_idx])

		#print y_score

		

		clf.fit(X[train_idx], y[train_idx])
		predicted = clf.predict(X[test_idx])



		accuracy = accuracy_score(y[test_idx], predicted)
		precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_3)
		macro_f1 = f1_score(y[test_idx], predicted, average='macro')

		accuracies.append(accuracy)
		precisions_pos.append(precision[0])
		precisions_neg.append(precision[1])
		precisions_neu.append(precision[2])
		recalls_pos.append(recall[0])
		recalls_neg.append(recall[1])
		recalls_neu.append(recall[2])
		f1_scores_pos.append(fscore[0])
		f1_scores_neg.append(fscore[1])
		f1_scores_neu.append(fscore[2])
		macro_f1_vec.append(macro_f1)

		#conf_matrix = confusion_matrix(y[test_idx], predicted, labels=[1, -1, 0])

		#print conf_matrix

		#sys.exit()

		"""
		n_classes = 3

		fpr = dict()
		tpr = dict()
		roc_auc = dict()


		print y[test_idx][y == 1]
		print "---------------------"
		print y_score[y[test_idx][y == 1].index.tolist(), 1]
		
		for i in range(n_classes):
		    fpr[i], tpr[i], _ = roc_curve(y[test_idx][y == i], y_score[y[test_idx][y == i].index.tolist(), i])
		    roc_auc[i] = auc(fpr[i], tpr[i])
		"""
		


	if label == "SVM":
		accuracies_svm = accuracies
		precisions_pos_svm = precisions_pos
		precisions_neg_svm = precisions_neg
		precisions_neu_svm = precisions_neu
		recalls_pos_svm = recalls_pos
		recalls_neg_svm = recalls_neg
		recalls_neu_svm = recalls_neu
		f1_scores_pos_svm = f1_scores_pos
		f1_scores_neg_svm = f1_scores_neg
		f1_scores_neu_svm = f1_scores_neu
		macro_f1_vec_svm = macro_f1_vec

	elif label == "MNB":
		accuracies_mnb = accuracies
		precisions_pos_mnb = precisions_pos
		precisions_neg_mnb = precisions_neg
		precisions_neu_mnb = precisions_neu
		recalls_pos_mnb = recalls_pos
		recalls_neg_mnb = recalls_neg
		recalls_neu_mnb = recalls_neu
		f1_scores_pos_mnb = f1_scores_pos
		f1_scores_neg_mnb = f1_scores_neg
		f1_scores_neu_mnb = f1_scores_neu
		macro_f1_vec_mnb = macro_f1_vec

	elif label == "DT":
		accuracies_dt = accuracies
		precisions_pos_dt = precisions_pos
		precisions_neg_dt = precisions_neg
		precisions_neu_dt = precisions_neu
		recalls_pos_dt = recalls_pos
		recalls_neg_dt = recalls_neg
		recalls_neu_dt = recalls_neu
		f1_scores_pos_dt = f1_scores_pos
		f1_scores_neg_dt = f1_scores_neg
		f1_scores_neu_dt = f1_scores_neu
		macro_f1_vec_dt = macro_f1_vec

	elif label == "LR":
		accuracies_lr = accuracies
		precisions_pos_lr = precisions_pos
		precisions_neg_lr = precisions_neg
		precisions_neu_lr = precisions_neu
		recalls_pos_lr = recalls_pos
		recalls_neg_lr = recalls_neg
		recalls_neu_lr = recalls_neu
		f1_scores_pos_lr = f1_scores_pos
		f1_scores_neg_lr = f1_scores_neg
		f1_scores_neu_lr = f1_scores_neu
		macro_f1_vec_lr = macro_f1_vec

	elif label == "NNET":
		accuracies_nnet = accuracies
		precisions_pos_nnet = precisions_pos
		precisions_neg_nnet = precisions_neg
		precisions_neu_nnet = precisions_neu
		recalls_pos_nnet = recalls_pos
		recalls_neg_nnet = recalls_neg
		recalls_neu_nnet = recalls_neu
		f1_scores_pos_nnet = f1_scores_pos
		f1_scores_neg_nnet = f1_scores_neg
		f1_scores_neu_nnet = f1_scores_neu
		macro_f1_vec_nnet = macro_f1_vec

	elif label == "Ensemble":
		accuracies_ens = accuracies
		precisions_pos_ens = precisions_pos
		precisions_neg_ens = precisions_neg
		precisions_neu_ens = precisions_neu
		recalls_pos_ens = recalls_pos
		recalls_neg_ens = recalls_neg
		recalls_neu_ens = recalls_neu
		f1_scores_pos_ens = f1_scores_pos
		f1_scores_neg_ens = f1_scores_neg
		f1_scores_neu_ens = f1_scores_neu
		macro_f1_vec_ens = macro_f1_vec



""" #################### Analytics - Print Performance Measures Results ########################### """

methods = ["SVM", "MNB", "DT", "LR", "NNET", "ENS"]
accuracy_results = [np.mean(accuracies_svm), np.mean(accuracies_mnb), np.mean(accuracies_dt), np.mean(accuracies_lr), np.mean(accuracies_nnet), np.mean(accuracies_ens)]

best_accuracy = "|".join(methods[i] for i in [i for i, x in enumerate(accuracy_results) if x == max(accuracy_results)])
worst_accuracy = "|".join(methods[i] for i in [i for i, x in enumerate(accuracy_results) if x == min(accuracy_results)])

print("Best accuracy are from: %s" % best_accuracy)
print("Worst accuracy are from: %s" % worst_accuracy)

prec_pos_results = [np.mean(precisions_pos_svm), np.mean(precisions_pos_mnb), np.mean(precisions_pos_dt), np.mean(precisions_pos_lr), np.mean(precisions_pos_nnet), np.mean(precisions_pos_ens)]
rec_pos_results = [np.mean(recalls_pos_svm), np.mean(recalls_pos_mnb), np.mean(recalls_pos_dt), np.mean(recalls_pos_lr), np.mean(recalls_pos_nnet), np.mean(recalls_pos_ens)]
f1_pos_results = [np.mean(f1_scores_pos_svm), np.mean(f1_scores_pos_mnb), np.mean(f1_scores_pos_dt), np.mean(f1_scores_pos_lr), np.mean(f1_scores_pos_nnet), np.mean(f1_scores_pos_ens)]

prec_neg_results = [np.mean(precisions_neg_svm), np.mean(precisions_neg_mnb), np.mean(precisions_neg_dt), np.mean(precisions_neg_lr), np.mean(precisions_neg_nnet), np.mean(precisions_neg_ens)]
rec_neg_results = [np.mean(recalls_neg_svm), np.mean(recalls_neg_mnb), np.mean(recalls_neg_dt), np.mean(recalls_neg_lr), np.mean(recalls_neg_nnet), np.mean(recalls_neg_ens)]
f1_neg_results = [np.mean(f1_scores_neg_svm), np.mean(f1_scores_neg_mnb), np.mean(f1_scores_neg_dt), np.mean(f1_scores_neg_lr), np.mean(f1_scores_neg_nnet), np.mean(f1_scores_neg_ens)]

prec_neut_results = [np.mean(precisions_neu_svm), np.mean(precisions_neu_mnb), np.mean(precisions_neu_dt), np.mean(precisions_neu_lr), np.mean(precisions_neu_nnet), np.mean(precisions_neu_ens)]
rec_neut_results = [np.mean(recalls_neu_svm), np.mean(recalls_neu_mnb), np.mean(recalls_neu_dt), np.mean(recalls_neu_lr), np.mean(recalls_neu_nnet), np.mean(recalls_neu_ens)]
f1_neut_results = [np.mean(f1_scores_neu_svm), np.mean(f1_scores_neu_mnb), np.mean(f1_scores_neu_dt), np.mean(f1_scores_neu_lr), np.mean(f1_scores_neu_nnet), np.mean(f1_scores_neu_ens)]

macro_f1_results = [np.mean(macro_f1_vec_svm), np.mean(macro_f1_vec_mnb), np.mean(macro_f1_vec_dt), np.mean(macro_f1_vec_lr), np.mean(macro_f1_vec_nnet), np.mean(macro_f1_vec_ens)]

best_macro_f1 = "|".join(methods[i] for i in [i for i, x in enumerate(macro_f1_results) if x == max(macro_f1_results)])
worst_macro_f1 = "|".join(methods[i] for i in [i for i, x in enumerate(macro_f1_results) if x == min(macro_f1_results)])

print("Best MacroF1 are from: %s" % best_macro_f1)
print("Worst MacroF1 are from: %s" % worst_macro_f1)


results_df = OrderedDict([
	('method'	,	methods),
   	('accuracy'	,	accuracy_results),

   	('prec_pos'	,	prec_pos_results),
   	('rec_pos'	,	rec_pos_results),
   	('f1_pos'	,	f1_pos_results),
   	
   	('prec_neg'	,	prec_neg_results),
   	('rec_neg'	,	rec_neg_results),
   	('f1_neg'	,	f1_neg_results),
   	
   	('prec_neut',	prec_neut_results),
   	('rec_neut'	,	rec_neut_results),
   	('f1_neut'	,	f1_neut_results),
   	
   	('macro_f1'	,	macro_f1_results)
])

df = pd.DataFrame(results_df)

df.to_csv(analytics_results_file_name, sep=',')

'''


""" ****************************************** 2 CLASSES ************************************************ """



for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['SVM', 'MNB', 'DT', 'LR', 'NNET', 'Ensemble']):

	accuracies = []
	precisions_pos = []
	precisions_neg = []
	recalls_pos = []
	recalls_neg = []
	f1_scores_pos = []
	f1_scores_neg = []
	macro_f1_vec = []


	# StratifiedKFold is a variation of k-fold which returns stratified folds: each set 
	#contains approximately the same percentage of samples of each target class as the complete set.
	skf = StratifiedKFold(n_splits=nfolds)

	for train_idx, test_idx in skf.split(X, y):



		# TODO Vai dar certo!!! So precisa ver como pegar probabilidade de predicao pra cada classe (y_score = clf.fit(X[train_idx], y[train_idx]).decision_function(X[test_idx]))

		#print y[test_idx]

		#print "=============================================="

		#print y[test_idx][y == 0]



		#sys.exit()

		

		#clf_temp = clf

		#y_score = clf_temp.fit(X[train_idx], y[train_idx]).decision_function(X[test_idx])

		#print y_score

		

		clf.fit(X[train_idx], y[train_idx])
		predicted = clf.predict(X[test_idx])



		accuracy = accuracy_score(y[test_idx], predicted)
		precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=labels_order_2)
		macro_f1 = f1_score(y[test_idx], predicted, average='macro')

		accuracies.append(accuracy)
		precisions_pos.append(precision[0])
		precisions_neg.append(precision[1])
		recalls_pos.append(recall[0])
		recalls_neg.append(recall[1])
		f1_scores_pos.append(fscore[0])
		f1_scores_neg.append(fscore[1])
		macro_f1_vec.append(macro_f1)

		#conf_matrix = confusion_matrix(y[test_idx], predicted, labels=[1, -1, 0])

		#print conf_matrix

		#sys.exit()

		"""
		n_classes = 3

		fpr = dict()
		tpr = dict()
		roc_auc = dict()


		print y[test_idx][y == 1]
		print "---------------------"
		print y_score[y[test_idx][y == 1].index.tolist(), 1]
		
		for i in range(n_classes):
		    fpr[i], tpr[i], _ = roc_curve(y[test_idx][y == i], y_score[y[test_idx][y == i].index.tolist(), i])
		    roc_auc[i] = auc(fpr[i], tpr[i])
		"""
		


	if label == "SVM":
		accuracies_svm = accuracies
		precisions_pos_svm = precisions_pos
		precisions_neg_svm = precisions_neg
		recalls_pos_svm = recalls_pos
		recalls_neg_svm = recalls_neg
		f1_scores_pos_svm = f1_scores_pos
		f1_scores_neg_svm = f1_scores_neg
		macro_f1_vec_svm = macro_f1_vec

	elif label == "MNB":
		accuracies_mnb = accuracies
		precisions_pos_mnb = precisions_pos
		precisions_neg_mnb = precisions_neg
		recalls_pos_mnb = recalls_pos
		recalls_neg_mnb = recalls_neg
		f1_scores_pos_mnb = f1_scores_pos
		f1_scores_neg_mnb = f1_scores_neg
		macro_f1_vec_mnb = macro_f1_vec

	elif label == "DT":
		accuracies_dt = accuracies
		precisions_pos_dt = precisions_pos
		precisions_neg_dt = precisions_neg
		recalls_pos_dt = recalls_pos
		recalls_neg_dt = recalls_neg
		f1_scores_pos_dt = f1_scores_pos
		f1_scores_neg_dt = f1_scores_neg
		macro_f1_vec_dt = macro_f1_vec

	elif label == "LR":
		accuracies_lr = accuracies
		precisions_pos_lr = precisions_pos
		precisions_neg_lr = precisions_neg
		recalls_pos_lr = recalls_pos
		recalls_neg_lr = recalls_neg
		f1_scores_pos_lr = f1_scores_pos
		f1_scores_neg_lr = f1_scores_neg
		macro_f1_vec_lr = macro_f1_vec

	elif label == "NNET":
		accuracies_nnet = accuracies
		precisions_pos_nnet = precisions_pos
		precisions_neg_nnet = precisions_neg
		recalls_pos_nnet = recalls_pos
		recalls_neg_nnet = recalls_neg
		f1_scores_pos_nnet = f1_scores_pos
		f1_scores_neg_nnet = f1_scores_neg
		macro_f1_vec_nnet = macro_f1_vec

	elif label == "Ensemble":
		accuracies_ens = accuracies
		precisions_pos_ens = precisions_pos
		precisions_neg_ens = precisions_neg
		recalls_pos_ens = recalls_pos
		recalls_neg_ens = recalls_neg
		f1_scores_pos_ens = f1_scores_pos
		f1_scores_neg_ens = f1_scores_neg
		macro_f1_vec_ens = macro_f1_vec



""" #################### Analytics - Print Performance Measures Results ########################### """

methods = ["SVM", "MNB", "DT", "LR", "NNET", "ENS"]
accuracy_results = [np.mean(accuracies_svm), np.mean(accuracies_mnb), np.mean(accuracies_dt), np.mean(accuracies_lr), np.mean(accuracies_nnet), np.mean(accuracies_ens)]

best_accuracy = "|".join(methods[i] for i in [i for i, x in enumerate(accuracy_results) if x == max(accuracy_results)])
worst_accuracy = "|".join(methods[i] for i in [i for i, x in enumerate(accuracy_results) if x == min(accuracy_results)])

print("Best accuracy are from: %s" % best_accuracy)
print("Worst accuracy are from: %s" % worst_accuracy)

prec_pos_results = [np.mean(precisions_pos_svm), np.mean(precisions_pos_mnb), np.mean(precisions_pos_dt), np.mean(precisions_pos_lr), np.mean(precisions_pos_nnet), np.mean(precisions_pos_ens)]
rec_pos_results = [np.mean(recalls_pos_svm), np.mean(recalls_pos_mnb), np.mean(recalls_pos_dt), np.mean(recalls_pos_lr), np.mean(recalls_pos_nnet), np.mean(recalls_pos_ens)]
f1_pos_results = [np.mean(f1_scores_pos_svm), np.mean(f1_scores_pos_mnb), np.mean(f1_scores_pos_dt), np.mean(f1_scores_pos_lr), np.mean(f1_scores_pos_nnet), np.mean(f1_scores_pos_ens)]

prec_neg_results = [np.mean(precisions_neg_svm), np.mean(precisions_neg_mnb), np.mean(precisions_neg_dt), np.mean(precisions_neg_lr), np.mean(precisions_neg_nnet), np.mean(precisions_neg_ens)]
rec_neg_results = [np.mean(recalls_neg_svm), np.mean(recalls_neg_mnb), np.mean(recalls_neg_dt), np.mean(recalls_neg_lr), np.mean(recalls_neg_nnet), np.mean(recalls_neg_ens)]
f1_neg_results = [np.mean(f1_scores_neg_svm), np.mean(f1_scores_neg_mnb), np.mean(f1_scores_neg_dt), np.mean(f1_scores_neg_lr), np.mean(f1_scores_neg_nnet), np.mean(f1_scores_neg_ens)]

macro_f1_results = [np.mean(macro_f1_vec_svm), np.mean(macro_f1_vec_mnb), np.mean(macro_f1_vec_dt), np.mean(macro_f1_vec_lr), np.mean(macro_f1_vec_nnet), np.mean(macro_f1_vec_ens)]

best_macro_f1 = "|".join(methods[i] for i in [i for i, x in enumerate(macro_f1_results) if x == max(macro_f1_results)])
worst_macro_f1 = "|".join(methods[i] for i in [i for i, x in enumerate(macro_f1_results) if x == min(macro_f1_results)])

print("Best MacroF1 are from: %s" % best_macro_f1)
print("Worst MacroF1 are from: %s" % worst_macro_f1)


results_df = OrderedDict([
	('method'	,	methods),
   	('accuracy'	,	accuracy_results),

   	('prec_pos'	,	prec_pos_results),
   	('rec_pos'	,	rec_pos_results),
   	('f1_pos'	,	f1_pos_results),
   	
   	('prec_neg'	,	prec_neg_results),
   	('rec_neg'	,	rec_neg_results),
   	('f1_neg'	,	f1_neg_results),
   	
   	('macro_f1'	,	macro_f1_results)
])

df = pd.DataFrame(results_df)

df.to_csv(analytics_results_file_name, sep=',')


# TODO - Create charts
# TODO - Perform 2 class experiments (create new .csv removing all neutral instances)
# TODO - Optmize Ensemble
# TODO - Vary pre-processing paramenters (collapse URL, hashtag, mentions, normalization of laughing patterns)
# TODO - For feature paper, vary pre-processing paramenters, beyond the ones in this paper, others such as greetings, emojis, hashtag interpretation...
# TODO - Remember meaning of each metric, also sensitivity, sensibility...
# TODO - Write on paper about StratifiedKFold
# TODO - LATER
	# understand linear kernel svm
	# understand more about args for nnet
	# improve ensemble with weighting voting, grid
	# try other ensemble methods

#for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['SVM', 'MNB', 'DT', 'Logistic Regression', 'NNET', 'Ensemble']):
#    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#print(dir(scores))
#print('vectorized %d tweets. found %d terms.' % (X.shape[0], X.shape[1]))








