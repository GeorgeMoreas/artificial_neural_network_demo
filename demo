import numpy as np
import matplotlib.pyplot as plt


def a():
    plt.cla()
    plt.clf()
    for x in xrange(1, 5):
        syn0 = 2*np.random.random((3, 1))-1
        print syn0
        plt.plot(syn0)
        plt.draw()
        plt.show(block=False)

def b():
    plt.cla()
    plt.clf()
    for x in xrange(1, 10):
        syn0 = 1/(1+np.exp(-x))
        syn1 = 1+np.exp(-x)
        print syn0
        print syn1
        plt.plot(x, syn0, 'bs', x, syn1, 'rs')
        plt.draw()
        plt.show(block=False)


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def clearGraph():
    plt.cla()
    plt.clf()


def c(iteration_count=10):
    clearGraph()

    # input dataset
    X = np.array([[0, 0, 1],   # 0
                  [0, 1, 1],   # 0
                  [1, 0, 1],   # 1
                  [1, 1, 1]])  # 1

    # output dataset
    y = np.array([[0, 0, 1, 1]]).T

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    #np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3, 1)) - 1

    fig = plt.figure()
    predict = fig.add_subplot(131)
    error = fig.add_subplot(132)
    #delta = fig.add_subplot(233)
    weight = fig.add_subplot(133)

    plt.show(block=False)

    for iter in xrange(iteration_count):

        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1, True)

        # update weights
        syn0 += np.dot(l0.T, l1_delta)

        print "l0:"
        print l0
        print ""
        print "l1:"
        print l1
        print ""

        # forward propagation
        predict.text(1, 0.95, 'green & yellow: lim -> 1', fontsize=10)
        predict.text(1, 0.05, 'red & blue: lim -> 0', fontsize=10)
        predict.axis([0, iter, 0, 1])
        predict.plot(iter, l1[0], 'rD', iter, l1[1], 'bD', iter, l1[2], 'gD', iter, l1[3], 'yD')

        # how much did we miss?
        error.text(1, 0.9, 'large circles:', fontsize=10)
        error.text(1, 0.8, 'error: lim -> 0', fontsize=10)
        error.axis([0, iter, -1, 1])
        error.plot(iter, l1_error[0], 'ro', iter, l1_error[1], 'bo', iter, l1_error[2], 'go', iter, l1_error[3], 'yo')

        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        error.text(1, -0.8, 'small diamonds:', fontsize=10)
        error.text(1, -0.9, 'delta: lim -> 0', fontsize=10)
        #delta.axis([0, iter, -0.2, 0.2])
        error.plot(iter, l1_delta[0], 'r.', iter, l1_delta[1], 'b.', iter, l1_delta[2], 'g.', iter, l1_delta[3], 'y.')

        # update weights
        weight.text(1, -3.0, 'weight: lim -> infinity', fontsize=10)
        weight.text(1, -3.5, 'magenta = 1', fontsize=10)
        weight.text(1, -4.0, 'cyan = 2', fontsize=10)
        weight.text(1, -4.5, 'black = 3', fontsize=10)
        weight.axis([0, iter, -5, 5])
        weight.plot(iter, syn0[0], 'ms', iter, syn0[1], 'cs', iter, syn0[2], 'ks')

        plt.draw()

    print "Output After Training:"
    print l1
