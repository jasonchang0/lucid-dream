import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn import utils, preprocessing, neighbors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
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
file = open('feature_rank.txt', 'w')
sys.stdout = file

filename = 'fs_master_dataset.csv.gz'
df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')
# print(df.head())

df = utils.shuffle(df, random_state=None, n_samples=len(df.index) // 10)

X = df.drop(['label'], axis=1, inplace=False)
X = X.astype(float)

y = df[['label']]
y = y.astype(int)

KNeighbors_classifier = neighbors.KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_jobs=-1)
QDA_classifier = QuadraticDiscriminantAnalysis()

'''
Naive Bayes classifier
posterior = prior occurrences * likelihood / evidence
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
# NuSVC_classifier = NuSVC()

combined_classifier = VotingClassifier(
    estimators=[('KNC', KNeighbors_classifier), ('QDA', QDA_classifier),
                ('Log', LogisticRegression_classifier),
                ('SGD', SGD_classifier), ('RF', RandomForest_classifier),
                ('SVC', SVC_classifier), ('LSVC', LinearSVC_classifier),
                ('DTC', DecisionTree_classifier),
                ('ABC', AdaBoost_classifier)], voting='hard', n_jobs=-1)

rfecv = RFECV(estimator=RandomForest_classifier, step=1, cv=StratifiedKFold(n_splits=3),
              scoring='accuracy', n_jobs=-1)

X = np.array(X).reshape(len(df.index), len(df.columns) - 1)
y = np.array(y).reshape(len(df.index), -1)

print(X.shape)
print(y.shape)

rfecv.fit(X, y)
print("Optimal number of features : {}".format(rfecv.n_features_))

# print(rfecv.support_)
print(rfecv.grid_scores_)

feature_rank = dict(zip(df.columns[1:], rfecv.ranking_))
feature_rank = sorted(feature_rank.items(), key=lambda kv: kv[1])
print(np.array(feature_rank)[:, 0])

# Serialize ranking of predictive features for further modeling
save_file = open('reduced_features.pickle', 'wb')
pickle.dump(np.array(feature_rank)[:, 0], save_file)
save_file.close()

sys.stdout = orig_stdout
file.close()

# Plot number of features VS. cross-validation scores
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linestyle="-", color='b', linewidth=5)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.show()
