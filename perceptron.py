import nengo
import math

model = nengo.Network()
with model:
    stim_a = nengo.Node(0)
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_a, a)

    stim_b = nengo.Node(0)
    b = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_b, b)
    
    c = nengo.Ensemble(n_neurons=200, dimensions=2, radius=2)
    nengo.Connection(a, c[0])
    nengo.Connection(b, c[1])
    d = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    e = nengo.Ensemble(n_neurons=200, dimensions=2, radius=2)
    nengo.Connection(a, e[0])
    nengo.Connection(b, e[1])
    f = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    g = nengo.Ensemble(n_neurons=200, dimensions=2, radius=2)
    nengo.Connection(d, g[0])
    nengo.Connection(f, g[1])
    h = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    def somme(x):
        w = [1,1]
        return x[0]*w[0] + x[1]*w[1]
        
    def sigmoid(a):
        return 1/(1+math.exp(-a))
        
    def perceptron(x):
        return sigmoid(somme(x))
        
    nengo.Connection(c, d, function=perceptron)
    nengo.Connection(e, f, function=perceptron)
    nengo.Connection(g, h, function=perceptron)