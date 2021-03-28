import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from opt import WAME
from sklearn.preprocessing import StandardScaler
import datetime


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
    scaler = StandardScaler()
    df['age'] = scaler.fit_transform(df['age'].values.reshape(-1, 1))
    df['hours_pw'] = scaler.fit_transform(df['hours_pw'].values.reshape(-1, 1))
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

    batch_size = 160
    train_ds = df_to_dataset(df, batch_size=batch_size)
    val_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)

    print(len(feature_columns))

    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(7, activation='relu'),
        #tf.keras.layers.Dense(7, activation='tanh'),
        #tf.keras.layers.Dense(7, activation='tanh'),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(#optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  #optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
                optimizer=WAME(learning_rate=0.0001),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.FalsePositives(),
                    keras.metrics.FalseNegatives(),
                    keras.metrics.TruePositives(),
                    keras.metrics.TrueNegatives()])

    # train/validate
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    # test
    test_df = get_clean_data(name='adult.test', skiprows=1, salary_str='<=50K.')
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=320)
    loss, accuracy, fp, fn, tp, tn = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    print("Loss", loss)
    print("False Pos", fp)
    print("False Neg", fn)
    print("True Pos", tp)
    print("True Neg", tn)
    from matplotlib import pyplot
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='cross-val')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['binary_accuracy'], label='train')
    pyplot.plot(history.history['val_binary_accuracy'], label='cross-val')
    pyplot.legend()
    pyplot.show()

