import numpy as np


def sigm(x):
    """
    This is the sigma function used in binary classification
    @param x: vector x for sigma argument
    @return: the calculated sigma of x
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    This is the softmax function used in multiclass classification
    @param x: vector x for softmax argument
    @return: the calculated softmax of x
    """
    exps = np.exp(x- np.max(x))
    return exps / np.sum(exps)


def softmax_der(x):
    """
    Calculates the derivate of the softmax
    @param x: for the parameter of derivative of softmax
    @return: derivative of softmax of x
    """
    return softmax(x) * (1 - softmax(x))


def sigm_deriv(z):
    """
    Calculates the derivate of the sigma function
    @param x: for the parameter of derivative of sigma
    @return: derivative of sigma of x
    """
    a = sigm(z)
    return a * (1 - a)


class MLP:

    def __init__(self, n, m, o, input_array, output_array):
        """

        @param n: number of neurons at input layer
        @param m: number of neurons at hidden layer
        @param o: number of neurons at output layer
        @param input_array: training data features
        @param output_array: training data labels
        """
        self.train_inputs = input_array
        self.train_outputs = output_array
        self.m = m
        self.n = n
        self.o = o

        if self.o == 1:
            self.activation = sigm
            self.der_activation = sigm_deriv
        else:
            self.activation = softmax
            self.der_activation = softmax_der

        np.random.seed(23)

        # hidden layer of n neurons
        self.w2 = np.random.randn(self.n, self.m)  # nxm
        self.b2 = np.random.randn(self.m, 1)

        # output layer has o neuron
        self.w3 = np.random.randn(self.m, self.o)
        self.b3 = np.random.randn(self.o, 1)

    def feedforward(self, xs):
        """
        This performs the feedforward function on the layer
        @param xs: input for the layer
        @return: output for the layer
        """
        a1 = np.array(xs).reshape(1, len(xs))
        z2 = ((self.w2).T.dot(a1.T) + self.b2).T
        a2 = self.activation(z2)
        z3 = ((self.w3.T).dot(a2.T) + self.b3).T
        a3 = self.activation(z3)
        return a3

    def backprop(self, xs, ys):
        """
        This performs the backpropagation on the weights
        @param xs: input training features
        @param ys:  input training labels
        @return: gives out the average of the bias and weights and costs
        """
        del_w2 = np.zeros(self.w2.shape, dtype=float)
        del_b2 = np.zeros(self.b2.shape, dtype=float)

        del_w3 = np.zeros(self.w3.shape, dtype=float)
        del_b3 = np.zeros(self.b3.shape, dtype=float)

        cost = 0.0

        for x, y in zip(xs, ys):
            a1 = np.array(x).reshape(1, len(x))
            z2 = (self.w2.T.dot(a1.T) + self.b2).T
            a2 = self.activation(z2)
            z3 = (self.w3.T.dot(a2.T) + self.b3).T
            a3 = self.activation(z3)

            delta3 = ((a3 - y) * self.der_activation(z3)).T
            delta2 = ((self.w3.dot(delta3)) * self.der_activation(z2.T))
            del_b3 += delta3
            del_w3 += (delta3.dot(a2)).T
            del_b2 += delta2

            del_w2 += (delta2.dot(a1)).T
            cost += ((a3 - y) ** 2).sum()

        n = len(ys)

        return del_b2 / n, del_w2 / n, del_b3 / n, del_w3 / n, cost / n

    def train(self, epochs, eta):
        """
        This performs the training of the model
        @param epochs: number of iteration till which the training ahas to be done
        @param eta: learning rate
        @return: list of costs for the iterations
        """
        xs = self.train_inputs
        ys = self.train_outputs
        cost = np.zeros((epochs,))

        for e in range(epochs):
            d_b2, d_w2, d_b3, d_w3, cost[e] = self.backprop(xs, ys)

            if e % 1000 == 0:
                eta = eta * 0.99

            self.b2 -= eta * d_b2
            self.w2 -= eta * d_w2
            self.b3 -= eta * d_b3
            self.w3 -= eta * d_w3
            print("Iteration:", e, "  loss: ", cost[e], "  with eta: ", eta)
        return cost

    def predict(self, xarr):
        """
        Predicts by using the trained model.
        @param xarr: list of features for which label is to be known
        @return:  list of labels
        """
        outarr = []
        for x in xarr:
            out = self.feedforward(x)
            outarr.append(np.argmax(out))
        return outarr
