import tensorflow as tf
import itertools
from tensorflow import keras
from matplotlib import pyplot
from census_data import get_clean_data, df_to_dataset, get_feature_columns
from wame_impl import WAME


#
# Adapted from
#       https://www.tensorflow.org/tutorials/structured_data/feature_columns
#       https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
#       https://keras.io/api/optimizers/
#

# this method generates the data/plot for comparing the optimisers in section 3
def optimiser_comparison():
    batch_size = 160
    network_width = 56
    activation = 'relu'
    dropout = .3

    # https://keras.io/api/optimizers/
    optimisers = {'WAME': WAME(learning_rate=0.0001),
                  'Adam': keras.optimizers.Adam(learning_rate=0.0001),
                  'RMSProp': keras.optimizers.RMSprop(learning_rate=0.0001),
                  'SGD': keras.optimizers.SGD(learning_rate=0.0001),
                  'Adagrad': keras.optimizers.Adagrad(learning_rate=0.0001)}

    df = get_clean_data()

    train_ds = df_to_dataset(df, batch_size=batch_size)
    val_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)

    models = {}

    for name, optimiser in optimisers.items():
        model = tf.keras.Sequential([
            keras.layers.DenseFeatures(get_feature_columns(df)),
            keras.layers.Dense(network_width, activation=activation),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=optimiser,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                               keras.metrics.Precision(name='precision'),
                               keras.metrics.Recall(name='recall'),
                               keras.metrics.AUC(name='auc')])

        models[name] = model

    trainings = {}
    testings = {}

    for name, model in models.items():
        # train/validate
        trainings[name] = model.fit(train_ds, validation_data=val_ds, epochs=20)
        # test
        test_df = get_clean_data(name='adult.test', skiprows=1, salary_str='<=50K.')
        test_ds = df_to_dataset(test_df, shuffle=False, batch_size=320)
        testings[name] = model.evaluate(test_ds, return_dict=True)

    plot_comparison(trainings)
    print(testings)


def plot_comparison(trainings):
    pyplot.rcParams['figure.figsize'] = (12, 10)
    pyplot.subplot(221)
    pyplot.title('Loss')
    for name, history in trainings.items():
        pyplot.plot(history.history['loss'], label=name + ' train')
    pyplot.legend()
    pyplot.subplot(222)
    pyplot.title('Loss CV')
    for name, history in trainings.items():
        pyplot.plot(history.history['val_loss'], label=name + ' cross-val')
    pyplot.legend()
    pyplot.subplot(223)
    pyplot.title('Accuracy')
    for name, history in trainings.items():
        pyplot.plot(history.history['accuracy'], label=name + ' train')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(224)
    pyplot.title('Accuracy CV')
    for name, history in trainings.items():
        pyplot.plot(history.history['val_accuracy'], label=name + ' cross-val')
    pyplot.legend()
    pyplot.show()


# this harness was adapted to test:
# 1. combinations of network architectures (Section 2.4 Network Parameters)
# 2. the stability results for WAME gathered in Appendix 1
def test_harness(runs=1):
    batch_sizes = [160]
    network_widths = [56]
    activations = ['relu']
    dropout = .3
    optimiser = WAME(learning_rate=0.00001)
    epochs = 20

    df = get_clean_data()

    combos = list(itertools.product(*[batch_sizes, network_widths, activations]))

    for batch_size, network_width, activation in combos:
        for _ in range(runs):
            trainings = {}
            train_ds = df_to_dataset(df, batch_size=batch_size)
            val_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)
            model = tf.keras.Sequential([
                keras.layers.DenseFeatures(get_feature_columns(df)),
                keras.layers.Dense(network_width, activation=activation),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=optimiser,
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                                   keras.metrics.Precision(name='precision'),
                                   keras.metrics.Recall(name='recall'),
                                   keras.metrics.AUC(name='auc')])

            # train/validate
            trainings['WAME'] = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

            def plot_metrics(history):
                pyplot.rcParams['figure.figsize'] = (12, 10)
                colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
                metrics = ['loss', 'auc', 'precision', 'recall']
                for n, m in enumerate(metrics):
                    name = m.replace("_", " ").capitalize()
                    pyplot.subplot(2, 2, n + 1)
                    pyplot.plot(history.epoch, history.history[m],
                                color=colors[7], label='Train')
                    pyplot.plot(history.epoch, history.history['val_' + m],
                                color=colors[8], linestyle="--", label='Val')
                    pyplot.xlabel('Epoch')
                    pyplot.ylabel(name)

                pyplot.legend()
                pyplot.show()

            # test
            test_df = get_clean_data(name='adult.test', skiprows=1, salary_str='<=50K.')
            test_ds = df_to_dataset(test_df, shuffle=False, batch_size=320)
            plot_metrics(trainings['WAME'])
            print(model.evaluate(test_ds, return_dict=True))
