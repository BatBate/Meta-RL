import tensorflow as tf
import numpy as np

action = tf.placeholder(dtype=tf.float32, shape=(None, 3))
label = tf.placeholder(dtype=tf.float32, shape=(None, 3))
with tf.variable_scope('encoder'):
    embedding_layer = tf.layers.dense(inputs=action, units=256, use_bias=False)
with tf.variable_scope('decoder'):
    out_action = tf.layers.dense(inputs=embedding_layer, units=3, use_bias=False)

x = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]
              ])

y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]
              ])


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

loss = tf.losses.mean_squared_error(label, out_action)

op = optimizer.minimize(loss)
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

init = tf.global_variables_initializer()

sess = tf.Session()


sess.run(init)

for i in range(10000):
    print(i)
    print(sess.run(loss, feed_dict={action: x, label: y}))
    sess.run(op, feed_dict={action: x, label: y})

emb, out_a = sess.run([embedding_layer, out_action], feed_dict={action: x, label: y})

saver.save(sess, './saved_embedding')
print(emb)
print(out_a)


