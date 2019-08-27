"""
Enabling eager execution changes how TensorFlow operations
behave—now they immediately evaluate and return their values
to Python. tf.Tensor objects reference concrete values instead
of symbolic handles to nodes in a computational graph. Since
there isn't a computational graph to build and run later in a
session, it's easy to inspect results using print() or a debugger.
Evaluating, printing, and checking tensor values does not break
the flow for computing gradients.

The tf.contrib.eager module contains symbols available to both eager
and graph execution environments and is useful for writing code to
work with graphs

For small graphs, eager execution runs a lot slower
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution() # start eager execution
#print(tf.executing_eagerly()) # ==> True


# try tfe.Variable
W_ih = tf.get_variable(name = "W_ih", initializer=tf.random_uniform([2,3], -1, 1))
W_ho = tf.get_variable(name = "W_ho", initializer=tf.random_uniform([3,1], -1, 1))
b_h = tf.get_variable(name = "b_h", initializer=tf.zeros([3]))
b_o = tf.get_variable(name = "b_o", initializer=tf.zeros([1]))

def nn(X):
    Z = tf.tanh(tf.matmul(X, W_ih) + b_h) # Hidden layer
    output = tf.sigmoid(tf.matmul(Z, W_ho) + b_o) # Output layer
    return output

def bin_xentropy(output, y):
    loss = tf.reduce_mean(((y * tf.log(output)) + ((1 - y) * tf.log(1.0 - output))) * -1)
    return loss


@tfe.implicit_value_and_gradients
def calc_gradient(X,y):
    return bin_xentropy(nn(X), y)

def preds(probs):
    bools = tf.greater_equal(probs, 0.5)
    return bools

def accuracy(preds,actual):
    acc = tf.reduce_mean(tf.cast(tf.equal(preds,tf.cast(actual,bool)), tf.float32))
    return float(acc)


X_train = tf.convert_to_tensor([[0,0],[0,1],[1,0],[1,1]], dtype=tf.float32)
y_train = tf.convert_to_tensor([[0],[1],[1],[0]], dtype=tf.float32)

optimizer = tf.train.GradientDescentOptimizer(0.1)

for epoch in range(2000+1):
    loss, grads_and_vars = calc_gradient(X_train, y_train)
    optimizer.apply_gradients(grads_and_vars)
    val_output = preds(nn(X_train)) # would be X_val
    val_acc = accuracy(val_output,y_train) # y_val
    if epoch%100==0:
        print("epoch: {}  loss: {}  val_acc: {}".format(epoch, loss.numpy(), val_acc))


#from timeit import default_timer as timer
#start = timer()
#end = timer()
#print(end-start)



"""
Computing gradients
Automatic differentiation is useful for implementing machine learning
algorithms such as backpropagation for training neural networks.
During eager execution, use tf.GradientTape to trace operations for computing gradients later.

tf.GradientTape is an opt-in feature to provide maximal performance when
not tracing. Since different operations can occur during each call,
all forward-pass operations get recorded to a "tape". To compute the
gradient, play the tape backwards and then discard. A particular
tf.GradientTape can only compute one gradient; subsequent calls throw
a runtime error.
"""


# ADD CLASSES!!!
