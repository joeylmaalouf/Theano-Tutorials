import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt


def floatX(X):
	return np.asarray(X, dtype = theano.config.floatX)


def init_weights(shape, std = 0.5):
	return theano.shared(floatX(np.random.randn(*shape) * std))


def sgd(cost, params, lr = 0.25):
	grads = T.grad(cost = cost, wrt = params)
	updates = []
	for p, g in zip(params, grads):
		updates.append([p, p - g * lr])
	return updates


def model(X, w_h, w_o):
	""" Do a dot product of weights and final hidden nodes
		to get the final output.
	"""
	# hidden layer with sigmoid activation
	h = T.nnet.sigmoid(T.dot(X, w_h))
	out = T.dot(h, w_o)
	return out


def hidden_out(X, w_h, w_o):
	""" Do elementwise multiplication of final hidden node * weight.
		This will give the final output values for each node at the final layer.
		These values, when added, give the final output.
	"""
	h = T.nnet.sigmoid(T.dot(X, w_h))
	# transpose to do elementwise multiply
	return h * w_o.T


trX = np.linspace(-2, 2, 100)
trY = np.exp(trX)
bias = np.ones(trX.shape[0])
trX = np.column_stack((bias, trX))

X = T.fvector(name = "X")
y = T.scalar(name = "y")

w_h = init_weights((2, 4))
w_o = init_weights((4, 1))

out= model(X, w_h, w_o)
hidden_out = hidden_out(X, w_h, w_o)

cost = T.mean((out-y)**2)
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs = [X, y], outputs = [cost, w_h], updates = updates, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = out, allow_input_downcast = True)
predict_hidden_out = theano.function(inputs = [X], outputs = hidden_out, allow_input_downcast = True)

for i in range(100):
	for X, y in zip(trX, trY):
		cost, w_h = train(X, y)
y_pred = [predict(x) for x in trX]

hidden_outputs = np.array([predict_hidden_out(x)[0] for x in trX])

plt.hold("on")
plt.plot(trX[:, 1], y_pred, label = "prediction")
plt.plot(trX[:, 1], trY, "k.", label = "data")

for node_j in range(hidden_outputs.shape[1]):
	plt.plot(trX[:, 1], hidden_outputs[:, node_j], label = "hidden node %d" % node_j)

plt.legend()
plt.title("Fn: y = e**x")
plt.show()
