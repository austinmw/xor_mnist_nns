"""
Still vanilla TensorFlow, but a more complex example including
single 3-node hidden layer with sigmoid activation

In addition:
- more sophisticated weight initialization
- layer function
- batch support (although not needed or really used for XOR)
- tensorboard logging
- saving best checkpoint
- continuing training from best checkpoint
- early stopping
"""

import os
import numpy as np
import tensorflow as tf
from  datetime import datetime

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

n_inputs = 2
n_hidden = 3
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        W = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=stddev), name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else: return Z

with tf.name_scope('nn'):
    hidden = neuron_layer(X, n_hidden, name='hidden', activation=tf.nn.sigmoid)
    probs = neuron_layer(hidden, n_outputs, name='outputs', activation=tf.nn.sigmoid)

with tf.name_scope('loss'):
    mse_loss = tf.reduce_mean(tf.squared_difference(y, probs), name='loss')
    loss_summary = tf.summary.scalar('mse_loss', mse_loss)
    # TRY WITH LOGITS THING!!
    #bin_xentropy_loss = tf.reduce_mean(( (y * tf.log(probs)) + ((1 - y) * tf.log(1.0 - probs)) ) * -1)

learning_rate = 1

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse_loss)

with tf.name_scope('eval'):
    correct = tf.equal(tf.greater_equal(probs,0.5), tf.cast(y,tf.bool))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
# The Saver class adds ops to save and restore variables to and from checkpoints
saver = tf.train.Saver()
logdir = log_dir("xor_nn")
# The FileWriter class provides a mechanism to create an event file in a given directory and add summaries and events to it
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 300
batch_size = 4

checkpoint_path = "/tmp/my_xor_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./final_model/my_xor_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_train = [[0],[1],[1],[0]]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = X_train
y_valid = y_train

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    for epoch in range(start_epoch,start_epoch+n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy,
                mse_loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            if epoch % 5 == 0:
                print("Epoch:", epoch,
                      "\tvalidation accuracy: {:.3f}%".format(accuracy_val*100),
                      "\tloss: {:.5f}".format(loss_val))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if loss_val < best_loss:
                    saver.save(sess, final_model_path)
                    best_loss = loss_val
                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break

X_test = X_train
y_test = y_train

# uncomment to restart training
os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

print(accuracy_val)
