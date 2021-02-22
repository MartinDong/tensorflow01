# 测试代码
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 不显示python使用过程中的警告
import warnings

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")
a = tf.constant(np.ones((3, 2)))
b = tf.constant(np.ones((2, 3)))
sess = tf.compat.v1.Session()
c = sess.run(tf.matmul(a, b))
print(c)
plt.plot(np.random.rand(20, 1))
plt.show()
