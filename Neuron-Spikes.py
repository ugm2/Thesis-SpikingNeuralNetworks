import nengo
import numpy as np

model = nengo.Network()
n_neurons = 25
with model:
    # our input node will output a sine wave with a period of 1 second
    a = nengo.Node(lambda t: np.sin(2 * np.pi * t), label='stimuli')

    b_rate = nengo.Ensemble(n_neurons, 1, 
            label='Rate Signal',
            neuron_type=nengo.RectifiedLinear(),
            seed=2)
    nengo.Connection(a, b_rate)

    # and another ensemble with spiking neurons
    b_spike = nengo.Ensemble(n_neurons, 1,
            label='LIF',
            neuron_type=nengo.LIF(), 
            seed=2)
    nengo.Connection(a, b_spike)
    
    c_spike = nengo.Ensemble(n_neurons, 1,
            label='Spiking Rectified Linear',
            neuron_type=nengo.SpikingRectifiedLinear(), 
            seed=2)
    nengo.Connection(a, c_spike)
    
    d_spike = nengo.Ensemble(n_neurons, 1,
            label='Adaptive LIF',
            neuron_type=nengo.AdaptiveLIF(), 
            seed=2)
    nengo.Connection(a, d_spike)
    
    e_spike = nengo.Ensemble(n_neurons, 1,
            label='Izhikevich',
            neuron_type=nengo.Izhikevich(), 
            seed=2)
    nengo.Connection(a, e_spike)
    
    