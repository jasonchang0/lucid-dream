import pandas as pd
import numpy as np
import glob
import os
import pickle
import gzip
import dask.dataframe as dd

os.chdir('../data')
# dir = './'

try:
    # Dataframes implement the Pandas API
    df = dd.read_csv('master_dataset.csv')

    # df = pd.read_csv('master_dataset.csv.gz', compression='gzip', nrows=100000)

    # open_file = open(dir + 'master_dataset.pickle', 'rb')
    # df = pickle.load(open_file)
    # open_file.close()

except FileNotFoundError as e:
    df = pd.DataFrame()

    for file in glob.glob('*.parquet'):
        if df.empty:
            df = pd.read_parquet(file, engine='pyarrow')
            continue

        df.append(pd.read_parquet(file, engine='pyarrow'))

    # save_file = open(dir + 'master_dataset.pickle', 'wb')
    # pickle.dump(df, save_file)
    # save_file.close()

    df.to_csv('master_dataset.csv.gz', index=False, compression='gzip', chunksize=100000)

print(df.head())
print(df.shape[0].compute())
