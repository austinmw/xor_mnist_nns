"""
vanilla TensorFlow XOR MLP example
3 hidden layers, tanh activation, sgd, lr=1

"""
import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[4,2], name = 'X')
y = tf.placeholder(tf.float32, shape=[4,1], name = 'y')

W_ih = tf.Variable(tf.random_uniform([2,3], -1, 1), name = "W_ih")
W_ho = tf.Variable(tf.random_uniform([3,1], -1, 1), name = "W_ho")

b_h = tf.Variable(tf.zeros([3]), name = "b_h")
b_o = tf.Variable(tf.zeros([1]), name = "b_o")

with tf.name_scope("layer2") as scope:
	Z = tf.tanh(tf.matmul(X, W_ih) + b_h)

with tf.name_scope("layer3") as scope:
	output = tf.sigmoid(tf.matmul(Z, W_ho) + b_o)

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(((y * tf.log(output)) + ((1 - y) * tf.log(1.0 - output))) * -1)

with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

X_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(2000+1):
    _, probs, loss = sess.run([train_step, output, cost], feed_dict={X: X_train, y: y_train})
    if i % 100 == 0:
        print('epoch %4d, loss: %.4f' % (i, loss), [round(p[0],4) for p in probs])
