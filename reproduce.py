from nnet import Node, Arc, makeNode, makeArc, makeNNet, connect, build
import random
import theano

def isEmpty(collection):
    return len(collection) == 0

def getArcs(nodelist):
    arcs = []
    for node in nodelist:
        arcs += node.incoming
    return arcs

def copyNode(node):
    n = makeNode(node.name, node.innov)
    n.innov = node.innov
    return n

def get_node_by_innov(nodelist, innov_number):
    for node in nodelist:
        if node.innov == innov_number:
            return node
    return None

def clone(nodelist):
    arclist = getArcs(nodelist)
    return _clone(nodelist, arclist)

def _clone(nodelist, arclist):
    progeny = []
    for node in nodelist:
        progeny.append(copyNode(node))

    for arc in arclist:
        parent_node = get_node_by_innov(progeny, arc.parent.innov)
        child_node = get_node_by_innov(progeny, arc.child.innov)
        makeArc(parent_node, child_node, weight=arc.W.eval(), innov=arc.innov)

    return progeny


def split_first(components):
    for i, c in enumerate(components):
        if c != components[0]:
            return components[:i], components[i:]
    return components, []

def mate(nodelists):
    components = []
    for nodelist in nodelists:
        components += nodelist
        components += getArcs(nodelist)

    nodelist = []
    arclist = []
    components = sorted(components)
    while not isEmpty(components):
        features, components = split_first(components)
        feature = random.choice(features)
        if type(feature) is Node:
            nodelist.append(feature)
        elif type(feature) is Arc:
            arclist.append(feature)
        else:
            assert False, "Features must be either nodes or arcs"

    return _clone(nodelist, arclist)

if __name__ == '__main__':
    a = makeNNet(1, 1, 1)
    b = clone(a)
    c = makeNNet(1, 1, 1)
    print 'a', a
    print 'b', b
    print 'c', c

    print 'a mated with b', mate([a, b])
    print 'a mated with c', mate([a, c])
