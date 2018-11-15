import os
import sys
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn import utils, preprocessing, neighbors, metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
import pickle


style.use('fivethirtyeight')

os.chdir('../data')


orig_stdout = sys.stdout
file = open('cross_validation.txt', 'w')
sys.stdout = file

open_file = open('reduced_features.pickle', 'rb')
feature_rank = pickle.load(open_file)
open_file.close()

filename = 'fs_master_dataset.csv.gz'
df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')
# print(df.head())


KNeighbors_classifier = neighbors.KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_jobs=-1)
QDA_classifier = QuadraticDiscriminantAnalysis()

'''
Naive Bayes classifier
posterior = prior occurrences * likelihood / evidence

*Requires non-negative X inputs
'''
# MNB_classifier = MultinomialNB()
# GaussianNB_classifier = GaussianNB()
# BernoulliNB_classifier = BernoulliNB()

DecisionTree_classifier = DecisionTreeClassifier()
LogisticRegression_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1)
SGD_classifier = SGDClassifier(penalty='elasticnet', learning_rate='optimal', n_jobs=-1)
RandomForest_classifier = RandomForestClassifier(n_estimators=50, n_jobs=-1)
AdaBoost_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.5)

'''
Support Vector Machines
'''
SVC_classifier = SVC(kernel='rbf', max_iter=1000)
LinearSVC_classifier = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=1000)
# NuSVC_classifier = NuSVC(nu=0.2, max_iter=1000)

# clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf = VotingClassifier(estimators=[('KNC', KNeighbors_classifier), ('QDA', QDA_classifier),
                                   ('Log', LogisticRegression_classifier),
                                   ('SGD', SGD_classifier), ('RF', RandomForest_classifier),
                                   ('SVC', SVC_classifier), ('LSVC', LinearSVC_classifier),
                                   ('DTC', DecisionTree_classifier),
                                   ('ABC', AdaBoost_classifier)], voting='hard', n_jobs=-1)

# Serialize classification model for future implementation
save_file = open('voting_classifier.pickle', 'wb')
pickle.dump(clf, save_file)
save_file.close()


accuracies = []
f1_scores = []

for _ in range(10):

    part_df = utils.shuffle(df, random_state=None, n_samples=250000)

    X = part_df.drop(['label'], axis=1, inplace=False)

    # Applying dimensional reduction from PCA
    X = X[feature_rank[:341]]
    X = X.astype(float)

    y = part_df[['label']]
    y = y.astype(int)

    X = np.array(X).reshape(len(part_df.index), 341)
    y = np.array(y).reshape(len(part_df.index), )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print('Accuracy:', accuracy)
    accuracies.append(accuracy)

    f1_score = metrics.f1_score(y_true=y_test, y_pred=clf.predict(X=X_test), average='weighted')
    print('F1-Score:', f1_score)
    f1_scores.append(f1_score)


print('Mean Accuracy: {}'.format(mean(accuracies)))
print('Mean F1-Score: {}'.format(mean(f1_scores)))


sys.stdout = orig_stdout
file.close()













