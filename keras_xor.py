"""
ADD ANOTHER FILE EAGER EXECUTION KERAS

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
import numpy as np
np.random.seed(42)

# try sigmoid,tanh,relu, try lr=0.1,0.01, try units=2,3,4
model = Sequential([
    Dense(input_dim=2,units=2, activation = 'relu'),
    Dense(units=1, activation ='sigmoid')
])

adam = Adam(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=adam)
#sgd = SGD(lr=3)
#model.compile(loss='binary_crossentropy', optimizer=sgd)

X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0],[1],[1],[0]], "float32")

model.fit(X, y, epochs=500, batch_size=4,verbose=1)

#print(model.summary())
print(model.predict_classes(X))

"""
Bengio has a paper on using sigmoids vs tanh in the context of a DNN.
The sigmoid activation function has the potential problem that it saturates
at zero and one while tanh saturates at plus and minus one. So if the activity
in the network during training is close to zero then the gradient for the
sigmoid activation function may go to zero (this is called "the vanishing
gradient problem")

Seems like best combos are:
1. Adam + relu
2. Adam + tanh
3. SGD + tanh
4. SGD + sigmoid
Worst:
5. Adam + sigmoid
6. SGD + relu
"""
