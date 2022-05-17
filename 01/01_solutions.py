# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd

# +
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
# -

df = pd.read_parquet('fhv_tripdata_2021-01.parquet')


def read_process_dataframe(filename):
    raw = pd.read_parquet(filename)

    raw['duration'] = raw.dropOff_datetime - raw.pickup_datetime
    raw.duration = raw.duration.apply(lambda td: td.total_seconds() / 60)

    raw['PUlocationID'] = raw.PUlocationID.fillna(-1)
    raw['DOlocationID'] = raw.DOlocationID.fillna(-1)
    
    raw[['PUlocationID', 'DOlocationID']] = raw[['PUlocationID', 'DOlocationID']].astype(str)
    raw['PU_DO'] = raw['PUlocationID'] + '_' + raw['DOlocationID']

    df = raw[(raw.duration >= 1) & (raw.duration <= 60)].copy(deep=True)
        
    return raw, df


raw, df = read_process_dataframe('fhv_tripdata_2021-01.parquet')

# + [markdown] tags=[]
# ### Q1
# -

raw.shape[0]

# ### Q2

round(raw.duration.mean(), 2)

# ### Q3

str(round(raw.groupby('PUlocationID')['PUlocationID'].count()[0] / raw.shape[0] * 100)) + '%'

# ### Q4

# +
categorical = ['PUlocationID', 'DOlocationID']

train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
X_train.shape[1]
# -

# ### Q5

# +
target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

round(mean_squared_error(y_train, y_pred, squared=False), 2)
# -

# ### Q6

# +
raw_val, df_val = read_process_dataframe('fhv_tripdata_2021-02.parquet')

categorical = ['PUlocationID', 'DOlocationID', 'PU_DO']

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

target = 'duration'
y_train = df[target].values
y_val = df_val[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

round(mean_squared_error(y_val, y_pred, squared=False), 2)
# -


