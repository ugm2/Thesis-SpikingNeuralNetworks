import numpy as np
import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise

# Create the model object
model = nengo.Network(label='Multiplication')
with model:
    # Create 4 ensembles of leaky integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, radius=10)
    B = nengo.Ensemble(100, dimensions=1, radius=10)
    combined = nengo.Ensemble(
        220, dimensions=2, radius=15)  # This radius is ~sqrt(10^2+10^2)
    prod = nengo.Ensemble(100, dimensions=1, radius=20)

# Comment out the line below for 'normal' encoders
combined.encoders = Choice([[1, 1], [-1, 1], [1, -1], [-1, -1]])

with model:
    # Create a piecewise step function for input
    inputA = nengo.Node(Piecewise({0: 0, 2.5: 10, 4: -10}))
    inputB = nengo.Node(Piecewise({0: 10, 1.5: 2, 3: 0, 4.5: 2}))

    correct_ans = Piecewise({0: 0, 1.5: 0, 2.5: 20, 3: 0, 4: 0, 4.5: -20})
    
    # Connect the input nodes to the appropriate ensembles
    nengo.Connection(inputA, A)
    nengo.Connection(inputB, B)

    # Connect input ensembles A and B to the 2D combined ensemble
    nengo.Connection(A, combined[0])
    nengo.Connection(B, combined[1])

    # Define a function that computes the multiplication of two inputs
    def product(x):
        return x[0] * x[1]

    # Connect the combined ensemble to the output ensemble D
    nengo.Connection(combined, prod, function=product)