import tensorflow as tf


class NN:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(7, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x, y):
        self.model.fit(x, y)
        print("Training", self.model.evaluate(x, y));

    def test_predictions(self, x):
        predictions = self.model.predict(x)
        return predictions

    def test_accuracy(self, x, y):
        accuracy = self.model.evaluate(x, y)
        print(accuracy)
        return accuracy
