import numpy as np
import nengo
from nengo.processes import WhiteSignal

model = nengo.Network()
with model:
    # -- input and pre popluation
    inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
    pre = nengo.Ensemble(120, dimensions=2)
    nengo.Connection(inp, pre)

    # -- post population
    post = nengo.Ensemble(60, dimensions=1)

    # -- reference population, containing the actual product
    product = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(
        inp, product, function=lambda x: x[0] * x[1], synapse=None)

    # -- error population
    error = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(post, error)
    nengo.Connection(product, error, transform=-1)

    # -- learning connection
    conn = nengo.Connection(
        pre,
        post,
        function=lambda x: np.random.random(1),
        learning_rule_type=nengo.PES())
    nengo.Connection(error, conn.learning_rule)

    # -- inhibit error after 40 seconds
    inhib = nengo.Node(lambda t: 2.0 if t > 40.0 else 0.0)
    nengo.Connection(inhib, error.neurons, transform=[[-1]] * error.n_neurons)