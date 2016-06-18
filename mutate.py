from nnet import node, connect, build
import random

def isEmpty(collection):
    return len(collection) == 0

def split(arc):
    arc.parent.outgoing.remove(self)
    arc.child.incoming.remove(self)
    node = Node()
    connect(arc.parent, node)
    connect(arc.child, node)

def split_arc(nodes):
    node = random.choice(nodes)
    arc = random.choice(node.incoming + node.outgoing)
    split(arc)

def get_ancestors(node):
    if isEmpty(node.incoming):
        return []
    else:
        arc = random.choice(node.incoming)
        return [arc.parent] + get_ancestors(arc.parent)

def get_descendants(node):
    if isEmpty(node.outgoing):
        return []
    else:
        arc = random.choice(node.outgoing)
        return [arc.child] + get_descendants(arc.child)

def get_ancestor_and_descendant(node):
    descendants = get_descendants(node)
    ancestors = get_ancestors(node)

    assert not isEmpty(ancestors) or not isEmpty(descendants)

    if isEmpty(descendants):
        descendants = [node]
    elif isEmpty(ancestors):
        ancestors = [node]

    return random.choice(ancestors), random.choice(descendants)

def add_node(nodes):
    source = random.choice(nodes)
    ancestor, descendant = get_ancestor_and_descendant(source)

    n = Node()
    connect(ancestor, n)
    connect(n, descendant)

def add_arc(nodes):
    source = random.choice(nodes)
    ancestor, descendant = get_ancestor_and_descendant(source)


if __name__ == '__main__':
    a = Node(name='a')
    b = Node(name='b')
    c = Node(name='c')
    d = Node(name='d')
    nodes = [a, b, c, d]
    connect(a, b)
    connect(b, c)
    connect(c, d)
    add_arc(nodes)
    print nodes
    add_Node(nodes)
    print nodes

    print 'descendants of a:', get_descendants(a)
    print 'ancestors of c:', get_ancestors(c)
