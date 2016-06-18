import theano.tensor as T
import theano.tensor.nnet as nnet
import theano
import numpy as np

def flatten(list_of_lists):
    result = []
    for l in list_of_lists:
        if type(l) is list:
            result += flatten(l)
        else:
            result.append(l)
    return result

def makeNNet(*layer_sizes):
    layers = []
    for layer_size in layer_sizes:
        layers.append([makeNode() for i in range(layer_size)])

    for parents, children in zip(layers[:-1], layers[1:]):
        for parent in parents:
            for child in children:
                connect(parent, child)

    return flatten(layers)

def makeNode(name=None, innov=None):
    return Node(name, innov)

def makeArc(parent_node, child_node, weight=None, innov=None):
    return Arc(parent_node, child_node, weight, innov)

def connect(parent, child):
    makeArc(parent, child)

def isEmpty(collection):
    return len(collection) == 0

def makeSharedVar(array):
    return theano.shared(np.asarray(array, dtype=theano.config.floatX))

INNOV_NUMBER = 0

def getNextInnov():
    global INNOV_NUMBER
    INNOV_NUMBER += 1
    return INNOV_NUMBER

class Node(object):
    def __init__(self, name=None, innov=None):
        self.incoming = []
        self.outgoing = []
        bias = np.random.randn()
        self.B = makeSharedVar(bias)
        if innov == None:
            innov = getNextInnov()
        self.innov = innov

        if name == None:
            self.name = 'innov=' + str(self.innov)
        else:
            self.name = name

    def __repr__(self):
        if self.name is not None:
            return 'node[' + str(self.name) + ']'
        else:
            return '<{0}.{1} object at {2}>'.format(self.__module__, type(self).__name__, hex(id(self)))
    def __cmp__(self, other):
        return cmp(self.innov, other.innov)
    def __eq__(self, other):
        return self.innov == other.innov

class Arc(object):
    def __init__(self, parent, child, weight=None, innov=None):
        self.parent = parent
        self.child = child
        self.parent.outgoing.append(self)
        self.child.incoming.append(self)

        if weight == None:
            weight = np.random.randn()
        self.W = makeSharedVar(weight)

        if innov == None:
            innov = getNextInnov()
        self.innov = innov

    def __cmp__(self, other):
        return cmp(self.innov, other.innov)
    def __eq__(self, other):
        return self.innov == other.innov

def build(nodes):
    inputs, outputs = setup_model(nodes)
    targets = [T.scalar() for i in outputs]
    params = []
    for node in nodes:
        for arc in node.incoming:
            params.append(arc.W)
        if not isEmpty(node.incoming):
            params.append(node.B)
    mse = sum([(y - t)**2 for y, t in zip(outputs, targets)])
    step_size = T.scalar()
    updates = [(p, p - step_size * T.grad(cost=mse, wrt=p)) for p in params]
    ev = theano.function(inputs=inputs, outputs=outputs)
    gr = theano.function(inputs=inputs+targets+[step_size], outputs=mse, updates=updates)

    def evaluate(inputs):
        return np.asarray(ev(*inputs))
    def train(inputs, targets, step_size):
        return gr(*inputs+targets+[step_size])

    return evaluate, train

def setup_model(nodes):
    inputs = []
    outputs = []
    for node in nodes:
        if isEmpty(node.incoming):
            node.Y = T.scalar()
            inputs.append(node.Y)
            node.visited = True
        else:
            node.visited = False
    for node in nodes:
        if isEmpty(node.outgoing):
            outputs.append(get_output(node))
    return inputs, outputs

def get_output(node):
    if node.visited:
        return node.Y
    else:
        node.visited = True
        weighted_sum = node.B
        for arc in node.incoming:
            weighted_sum += get_output(arc.parent) * arc.W
        node.Y = node.activation(weighted_sum)
        return node.Y


# Example
if __name__ == "__main__":
    print "Attempting to learn the XOR function:"
    # The XOR function
    data = [
            ([0, 0], [0]), # 00 gets mapped onto 0
            ([0, 1], [1]), # 01 gets mapped onto 1
            ([1, 0], [1]), # 10 gets mapped onto 1
            ([1, 1], [0]), # 11 gets mapped onto 0
    ]

    for inputs, targets in data:
        print inputs, '-->', targets

    print "Initializing model."

    layer1 = [Node() for i in range(2)]
    layer2 = [Node() for i in range(3)]
    layer3 = [Node() for i in range(2)]

    for p in layer1:
        for c in layer2:
            connect(p, c)

    for p in layer2:
        for c in layer3:
            connect(p, c)

    # A hyper parameter, somewhat arbitrarily chosen. You have to experiment to see what step size works best.
    step_size = 5

    # We pass a list of all the nodes into the build method, to compile two theano functions for
    # evaluating and training the neural network represented by these nodes.
    evaluate, train = build(layer1+layer2+layer3)

    print "Training model."

    for i in range(1000):
        error = 0
        for inputs, targets in data:
            error += train(inputs, targets, step_size)
        if i % 100 == 0:
            print 'iteration', i, 'error:', error

    # Define a method to deprocess the outputs of the neural network from a list of floats to a list of integers
    def deprocess(outputs):
        for i, output in enumerate(outputs):
            outputs[i] = int(round(output))
        return outputs

    print "Testing model."

    for inputs, targets in data:
        outputs = evaluate(inputs)
        print inputs, outputs, targets, deprocess(outputs) == targets
