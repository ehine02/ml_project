import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from opt import WAMEFromRMSProp, WAMEOptimizer
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
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
    df['target'] = np.where(df['salary'] == salary_str, 0, 1)
    df = df.drop(columns=['workclass', 'fnlwgt', 'education_num', 'marital_status', 'relationship',
                          'capital_gain', 'capital_loss', 'salary'])
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
    print('Loaded data shape:', df.shape)
    print(df.head())
    return df


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)).batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds


if __name__ == '__main__':
    df = get_clean_data()

    # numeric columns
    feature_columns = [feature_column.numeric_column('age'), feature_column.numeric_column('hours_pw')]

    # categorical columns
    category_columns = ['education_lvl', 'occupation', 'race', 'sex', 'native_country']
    for col_name in category_columns:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, df[col_name].unique())
        category_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(category_column)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(df, batch_size=batch_size)
    val_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(14, activation=keras.activations.relu),
        layers.Dense(14, activation=keras.activations.relu),
        layers.Dropout(.1),
        layers.Dense(1, activation=keras.activations.sigmoid)
    ])

    model.compile(optimizer=WAMEOptimizer(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train/validate
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # test
    test_df = get_clean_data(name='adult.test', skiprows=1, salary_str='<=50K.')
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)



