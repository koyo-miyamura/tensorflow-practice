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

def print_score(metrics_names, score):
    for i in range(len(metrics_names)):
        print('{}: {}'.format(metrics_names[i], str(score[i])))

# 現状では入力画像は1枚
def img_preprocessing(input_img):
    input_img = input_img.convert('L') # グレースケール
    img_data  = 1.0 - np.asarray(input_img, dtype="float64") / 255 # mnistは黒背景なので白黒反転させる
    img_data  = np.array([img_data]) # model.predict できるように修正
    return img_data

def print_result(result):
    print('#######Result#######')
    print('number| probability')
    print('--------------------')
    for i in range(len(result[0])):
        print('{}| {:.8f}'.format(i, float(result[0][i])))
    print('prediction：{}'.format(np.argmax(result)))

def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = model_factory(x_train, y_train)

    score = model.evaluate(x_test, y_test)
    print_score(metrics_names = model.metrics_names, score = score)

    img_path  = "images/three.png"
    input_img = Image.open(img_path)
    img_data  = img_preprocessing(input_img)

    result = model.predict(img_data)
    print_result(result)

if __name__ == '__main__':
    main()
