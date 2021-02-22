import tensorflow as tf

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
hello = tf.constant("Hello tensorflow")

session = tf.compat.v1.Session()

print(session.run(hello))
