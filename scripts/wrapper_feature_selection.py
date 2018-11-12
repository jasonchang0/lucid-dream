import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn import utils, preprocessing
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

filename = 'fs_master_dataset.csv.gz'
df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')

df = utils.shuffle(df, random_state=None, n_samples=None)

X = df.drop(['label'], axis=1, inplace=False)
y = df[['label']]


'''
Naive Bayes classifier
posterior = prior occurrences * likelihood / evidence
'''
MNB_classifier = MultinomialNB()
GaussianNB_classifier = GaussianNB()
BernoulliNB_classifier = BernoulliNB()

DecisionTree_classifier = DecisionTreeClassifier()
LogisticRegression_classifier = LogisticRegression(n_jobs=-1)
SGD_classifier = SGDClassifier(penalty='elasticnet', n_jobs=-1, learning_rate='optimal')
RandomForest_classifier = RandomForestClassifier(n_estimators=50)
AdaBoost_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.5)


'''
Support Vector Machines
'''
SVC_classifier = SVC()
LinearSVC_classifier = LinearSVC()
NuSVC_classifier = NuSVC()

combined_classifier = VotingClassifier(estimators=[('MNB', MNB_classifier), ('GNB', GaussianNB_classifier),
                                                   ('BNB', BernoulliNB_classifier),
                                                   ('Log', LogisticRegression_classifier),
                                                   ('SGD', SGD_classifier), ('RF', RandomForest_classifier),
                                                   ('SVC', SVC_classifier), ('LSVC', LinearSVC_classifier),
                                                   ('NuSVC', NuSVC_classifier, ('DTC', DecisionTree_classifier),
                                                    'ABC', AdaBoost_classifier)], voting='hard')

rfecv = RFECV(estimator=combined_classifier, step=1, cv=StratifiedKFold(n_splits=2, shuffle=True),
              scoring='accuracy', n_jobs=-1)

rfecv.fit(np.array(X).reshape(len(df.index), len(df.columns)), np.array(y))
print("Optimal number of features : {}".format(rfecv.n_features_))

print(rfecv.support_)
print(rfecv.ranking_)


# Plot number of features VS. cross-validation scores
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linestyle="-", color='b', linewidths=5)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.show()






