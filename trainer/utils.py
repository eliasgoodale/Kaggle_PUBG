import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

tanh_scalar = MinMaxScaler(feature_range=(-1, 1))

def load_data(df, target, drop_cols):
    train_y = df[target]
    train_X = df.drop([*drop_cols,target], axis=1)

    return train_test_split(train_X, train_y, test_size=0.1, random_state=42)

def scale_features(df, cols, scalar):
    df[cols] = scalar.fit_transform(df[cols])
    return df

def reduce_mem(df):

    start_mem = round((df.memory_usage().sum() / 1024**2), 2)
    print(f'Memory usage of dataframe is {start_mem}')

    for col in df.columns:
        col_type = df[col].dtype
    
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = round((df.memory_usage().sum() / 1024**2), 2)
    print(f'Memory usage after optimization is: {end_mem}')

    saved_mem_perc = 100 * ((start_mem - end_mem) / start_mem)
    print(f'Decreased by {saved_mem_perc}%')

    return df