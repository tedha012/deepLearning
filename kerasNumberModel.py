from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(360, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="SGD",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=5,
    verbose=1,
)

model.save("mnist_keras_model.h5")
