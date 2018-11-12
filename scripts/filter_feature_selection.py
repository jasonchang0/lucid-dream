import os
import pandas as pd
import numpy as np
from sklearn import feature_selection as fs
from sklearn import preprocessing, utils
from multiprocessing import Process


os.chdir('../data')

filename = 'master_dataset.csv.gz'

df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')

x = df.drop(['label'], axis=1, inplace=False)

x.dropna(axis=0, inplace=True)
print('Total Samples:', len(x.index))

scaler = preprocessing.StandardScaler()

x = pd.DataFrame(data=scaler.fit_transform(x).reshape(-1, len(x.columns)),
                 columns=x.columns)

y = df[['label']]

"""
Select a threshold and remove features that have a variance lower than that threshold.
"""


def filter_by_variance(data, threshold=0.8 * (1-0.8)):
    var_filter = fs.VarianceThreshold(threshold)
    var_filter.fit_transform(data)


"""
The feature-feature correlation matrix is computed,
and pairs with a correlation exceeding a given 
threshold are identified iteratively.
"""


def filter_by_correlated_features(data):
    numeric_data = data

    feature_feature_correlation = numeric_data.corr()

    for feature in numeric_data.columns:
        if abs(np.mean(feature_feature_correlation[feature])) > 0.95:
            print('Excluded:', feature)
            data.drop(feature, axis=1, inplace=True)


filter_by_variance(x)
print('Filtered by variance.')

filter_by_correlated_features(x)
print('Filtered by correlated features.')

x = pd.concat([y, x], axis=1)

try:
    x.drop(['Unnamed: 0'], axis=1, inplace=True)
except (ValueError, KeyError) as e:
    pass

x = utils.shuffle(x, random_state=None, n_samples=None)
sub_x = utils.shuffle(x, random_state=None, n_samples=len(x.index)//10)

x.to_csv('fs_' + filename, index=False, compression='gzip', chunksize=10000)
sub_x.to_csv('fs_' + filename.replace('master', 'sample'), index=False, compression='gzip')







