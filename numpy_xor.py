import numpy as np
np.random.seed(1)

relu = lambda x: x.clip(min=0)
relu2deriv = lambda x: x>0

xor = np.array( [[ 1, 0 ],
                 [ 0, 1 ],
                 [ 0, 0 ],
                 [ 1, 1 ] ] )

labels = np.array([[ 1, 1, 0, 0]]).T
    
alpha = 0.1
hidden_size = 6

weights_0_1 = 2*np.random.random((2,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

for iteration in range(40):
    layer_2_error = 0
    for i in range(len(xor)):
        layer_0 = xor[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        layer_2_error += np.sum((layer_2 - labels[i:i+1]) ** 2)

        layer_2_delta = (layer_2 - labels[i:i+1])
        layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

        if(iteration % 10 == 9):
            print("Error:" + str(layer_2_error))