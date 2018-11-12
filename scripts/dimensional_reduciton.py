import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing, utils

os.chdir('../data')

filename = 'fs_master_dataset.csv.gz'
df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')

x = df.drop(['label'], axis=1, inplace=False)
y = df[['label']]

pca = PCA(n_components=0.95, copy=True, svd_solver='full')
pca.fit(x)

print('Number of Reduced Components:', pca.n_components_)
print(pca.explained_variance_ratio_)
print('Estimated Coverage:', sum(pca.explained_variance_ratio_[0:pca.n_components_]))

reduced_col = ['f{}'.format(_) for _ in range(pca.n_components_)]

pc_x = pca.transform(x)
pc_x = pd.DataFrame(data=pc_x, columns=reduced_col)

x = pd.concat([y, pc_x], axis=1)

try:
    x.drop(['Unnamed: 0'], axis=1, inplace=True)
except (ValueError, KeyError) as e:
    pass

x = utils.shuffle(x, random_state=None, n_samples=None)

x.to_csv(filename.replace('fs_master', 'reduced_dim'), index=False, compression='gzip', chunksize=10000)



