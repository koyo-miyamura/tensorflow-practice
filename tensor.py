import tensorflow as tf
import numpy as np
import h5py
import os
from PIL import Image

def model_factory(x_train, y_train):
    # Case: Model exist
    path = "./my_model.h5"
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        return model

    # Model create 
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.save('my_model.h5')
    return model

def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = model_factory(x_train, y_train)
    score = model.evaluate(x_test, y_test)
    for i in range(len(model.metrics_names)):
        print('{}: {}'.format(model.metrics_names[i], str(score[i])))

    input_img = Image.open("images/three.png")
    input_img = input_img.convert('L')
    predict   = 1.0 - np.asarray(input_img, dtype="float64") / 255 # mnistは黒背景
    predict   = np.array([predict])
    res = model.predict(predict)
    print(res)
    print(np.argmax(res))

if __name__ == '__main__':
    main()
