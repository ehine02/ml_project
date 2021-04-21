import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import feature_column
from sklearn.preprocessing import StandardScaler

#
# Adapted from
#       https://www.tensorflow.org/tutorials/structured_data/feature_columns
#       https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
#       https://keras.io/api/optimizers/
#


def get_clean_data(name='adult.data', skiprows=0, salary_str='<=50K'):
    columns = ['age', 'workclass', 'fnlwgt', 'education_lvl', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_pw', 'native_country', 'salary']

    df = pd.read_csv(name, header=None, names=columns, skipinitialspace=True, skiprows=skiprows)
    df = df.dropna()

    # Handle missing values by assigning the most common class in each case
    df['workclass'] = np.where(df['workclass'] == '?', 'Private', df['workclass'])
    df['occupation'] = np.where(df['occupation'] == '?', 'Prof-specialty', df['occupation'])
    df['native_country'] = np.where(df['native_country'] == '?', 'United-States', df['native_country'])

    # set the target salary band
    df['target'] = np.where(df['salary'] == salary_str, 0, 1)
    df = df.drop(columns=['salary'])

    # set the categorical columns
    df['education_lvl'] = pd.Categorical(df['education_lvl'])
    df['education_lvl'] = df.education_lvl.cat.codes
    df['occupation'] = pd.Categorical(df['occupation'])
    df['occupation'] = df.occupation.cat.codes
    df['race'] = pd.Categorical(df['race'])
    df['race'] = df.race.cat.codes
    df['sex'] = pd.Categorical(df['sex'])
    df['sex'] = df.sex.cat.codes
    df['native_country'] = pd.Categorical(df['native_country'])
    df['native_country'] = df.native_country.cat.codes
    df['workclass'] = pd.Categorical(df['workclass'])
    df['workclass'] = df.workclass.cat.codes
    df['marital_status'] = pd.Categorical(df['marital_status'])
    df['marital_status'] = df.marital_status.cat.codes
    df['relationship'] = pd.Categorical(df['relationship'])
    df['relationship'] = df.relationship.cat.codes

    # scale the numeric columns
    scaler = StandardScaler()
    df['age'] = scaler.fit_transform(df['age'].values.reshape(-1, 1))
    df['hours_pw'] = scaler.fit_transform(df['hours_pw'].values.reshape(-1, 1))
    df['capital_gain'] = scaler.fit_transform(df['capital_gain'].values.reshape(-1, 1))
    return df


def get_feature_columns(df):
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns

    # numeric columns
    feature_columns = [feature_column.numeric_column('age'),
                       feature_column.numeric_column('hours_pw'),
                       feature_column.numeric_column('capital_gain')]

    # categorical columns
    category_columns = ['education_lvl', 'occupation', 'race', 'sex',
                        'native_country', 'workclass', 'marital_status', 'relationship']

    # set up the one-hot encoding for categorical columns
    for col_name in category_columns:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, df[col_name].unique())
        category_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(category_column)

    return feature_columns


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()

    # Oversampling performed here
    # first count the records of the majority
    majority_count = df['target'].value_counts().max()
    working = [df]
    # group by each salary band
    for _, salary_band in df.groupby('target'):
        # append N samples to working list where N is the difference between majority and this band
        working.append(salary_band.sample(majority_count - len(salary_band), replace=True))
    # add the working list contents to the overall dataframe
    df = pd.concat(working)
    print(df['target'].value_counts())

    # create the TF2 dataset from the raw dataframe
    labels = df.pop('target')

    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)).batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds
