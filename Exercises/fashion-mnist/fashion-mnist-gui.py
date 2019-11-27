import nengo
import nengo_dl
import numpy as np
from nengo_extras.gui import image_display_function
import tensorflow as tf


(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_test = x_test.reshape((x_test.shape[0], -1))

labelNames = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# Visual input process parameters
presentation_time = 0.25

model = nengo.Network()
with model:
    
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    # this is an optimization to improve the training speed,
    # since stateful behaviour is not required here
    nengo_dl.configure_settings(stateful=False)
    
    # Visual input (the MNIST images) to the network
    input_node = nengo.Node(
        nengo.processes.PresentInput(x_test, presentation_time), label="input"
    )
    
    x = nengo_dl.Layer(tf.keras.layers.Dense(64))(input_node)
    x = nengo_dl.Layer(neuron_type)(x)
    
    output_node = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)
    
    # Output display node
    #output_node = nengo.Node(size_in=10, label="output class")
    
    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=input_node.size_out)
    nengo.Connection(input_node, display_node, synapse=None)
    
    vocab_vectors = np.eye(len(labelNames))
    
    vocab = nengo.spa.Vocabulary(len(labelNames))
    for name, vector in zip(labelNames, vocab_vectors):
        vocab.add(name, vector)
        
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output_spa = nengo.spa.State(len(labelNames), subdimensions=10, vocab=vocab)
    nengo.Connection(output_node, output_spa.input)