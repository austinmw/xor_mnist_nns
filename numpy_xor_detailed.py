# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

import numpy as np
np.random.seed(42)

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 lr, activation='sigmoid'):
        self.epoch_count = 0
        self.activation = activation
        self.sigmoid = lambda x: 1 / (1+np.exp(-x))
        if activation == 'tanh': self.nonlinearity = lambda x: np.tanh(x)
        elif activation == 'relu': self.nonlinearity = lambda x: x * (x > 0)
        else: self.nonlinearity = self.sigmoid
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.wih = np.random.normal(0.0, self.xavier_he_init_std(self.inodes,self.hnodes), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, self.xavier_he_init_std(self.inodes,self.hnodes), (self.onodes, self.hnodes))
        self.bo = np.zeros((self.onodes,1))
        self.bh = np.zeros((self.hnodes,1))
        self.lr = lr

    def xavier_he_init_std(self,n_in,n_out):
        if self.activation == 'tanh':
            return np.sqrt(4)*np.sqrt(2/(n_in+n_out))
        elif self.activation == 'relu':
            return np.sqrt(2)*np.sqrt(2/(n_in+n_out))
        else: # sigmoid
            return np.sqrt(2/(n_in+n_out))

    def predict(self,X):
        self.forward(X)
        bin_thresh = 0.5
        probs = self.final_outputs.T
        class_predictions = [1 if i > bin_thresh else 0 for i in probs]
        return class_predictions

    def fit(self, X, y, verbose=True):
        self.forward(X)
        self.backward(y)
        self.epoch_count += 1
        if verbose and (self.epoch_count%100==0):
            print("epoch %4d mse loss:" % self.epoch_count, self.mse_loss)

    def forward(self, X, output=False):
        self.X = np.array(X).T
        hidden_inputs = np.dot(self.wih, self.X) + self.bh # broadcast biases
        self.hidden_outputs = self.nonlinearity(hidden_inputs)
        final_inputs = np.dot(self.who, self.hidden_outputs) + self.bo
        self.final_outputs = self.sigmoid(final_inputs)
        if output: return self.final_outputs

    def backward(self, y):
        y = np.array(y).T
        self.mse_loss = np.square((y-self.final_outputs)).mean()

        # CHAIN RULE FOR WEIGHT UPDATES
        # Hidden-Output layer weights update
        # =================================================================
        # Need to calculate the partial derivative of the loss with respect
        # to each weight in the layer Who
        # partial derivative of the error with respect to the output ∂E/∂O:
        # (1/n)*(targets-predictions)^2 -> (2/n)*(target-actual)*-1 -> (-2/n)*sum(target-actual)
        dEdO_Who = -2*(y-self.final_outputs)
        # partial derivative of the output with respect to the net input ∂O/∂X:
        # 1 / (1 + e^(-N)) -> sigmoid(N)*(1-sigmoid(N)) -> O*(1-O)
        dOdN_Who = self.final_outputs * (1-self.final_outputs)
        # partial derivative of the net inputs with respect to the weights ∂N/∂W:
        # Net_ho = W1*X1+W2+X2... -> X1,X2,...
        dNdW_Who = self.hidden_outputs
        # multiplying each part of the chain rule together:
        dEdW_Who = np.dot((dEdO_Who * dOdN_Who),  dNdW_Who.T) # dot b.c. of batch inputs
        # update the weights by multiplying by the learning rate
        nabla_Who = self.lr * dEdW_Who
        nabla_bo = self.lr * np.dot((dEdO_Who * dOdN_Who), np.expand_dims(np.ones(y.shape),axis=1))

        # Input-Hidden layer weights update: backpropagating errors
        # =================================================================
        # Need to calculate the partial derivative of the loss with respect
        # to each weight in the layer wih
        # similar but slightly different process because the output of each hidden
        # layer neuron contributes to the output/error of multiple output neurons
        # For Example, ∂E/∂Wih_1 = sum_over_outputs(∂E/∂out_o ∂out_o/∂net_o . . .
        # * ∂net_o/∂out_h) * ∂out_h/∂net_h * ∂net_h/∂wih_1
        # This equals sum(δ_o * Who) * out_h * (1 - out_h) * i_1
        # where δ_o = ∂E/∂net = ∂E/∂out * ∂out/∂net
        # errors from hidden layer split output error proportionally by weight contributions
        dEdO_Wih = np.dot(self.who.T, dEdO_Who) # first three p.d. terms
        dOdN_Wih = self.hidden_outputs * (1-self.hidden_outputs)
        dNdW_Wih = self.X # inputs
        dEdW_Wih = np.dot((dEdO_Wih * dOdN_Wih), dNdW_Wih.T)
        nabla_Wih = self.lr * dEdW_Wih
        nabla_bh = self.lr * np.dot((dEdO_Wih * dOdN_Wih), np.expand_dims(np.ones(y.shape),axis=1))

        # update the weights and biases
        self.who -= nabla_Who
        self.wih -= nabla_Wih
        self.bo -= nabla_bo
        self.bh -= nabla_bh

def main():

    X_train = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]
    y_train = [0,1,1,0]

    sz = lambda l: 1 if (isinstance(l[0], int) or isinstance(l[0], float)) else len(l[0])
    input_nodes = sz(X_train)
    output_nodes = sz(y_train)

    hidden_nodes = 2
    epochs = 500
    lr=0.1

    nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,lr,activation='tanh')

    # train
    for _ in range(epochs):
        nn.fit(X_train, y_train,verbose=True)
    # predict
    print(nn.predict(X_train))

if __name__ == "__main__":
    main()
