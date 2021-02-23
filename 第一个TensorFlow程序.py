# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
# 支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import numpy as np
# 画图的函数
import matplotlib.pyplot as plt

print(tf.__version__)

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=100)

print(model.predict([10.0]))
