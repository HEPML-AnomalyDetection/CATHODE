import tensorflow as tf
import yaml


def build_classifier(filename):
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)

    model = tf.keras.Sequential()
    for nodes in params['layers']:
        model.add(tf.keras.layers.Dense(nodes, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(params['learning_rate']))

    model.compile(loss=params['loss'],
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = build_classifier('classifier.yml')
    x = tf.random.uniform((1, 5))
    print(model(x))
