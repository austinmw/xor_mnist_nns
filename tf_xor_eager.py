"""
Enabling eager execution changes how TensorFlow operations
behave—now they immediately evaluate and return their values
to Python. tf.Tensor objects reference concrete values instead
of symbolic handles to nodes in a computational graph. Since
there isn't a computational graph to build and run later in a
session, it's easy to inspect results using print() or a debugger.
Evaluating, printing, and checking tensor values does not break
the flow for computing gradients.
"""
import tensorflow as tf

tf.enable_eager_execution() # start eager execution
print(tf.executing_eagerly()) # ==> True


"""
The tf.contrib.eager module contains symbols available to both eager
and graph execution environments and is useful for writing code to
work with graphs
"""
tfe = tf.contrib.eager



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
