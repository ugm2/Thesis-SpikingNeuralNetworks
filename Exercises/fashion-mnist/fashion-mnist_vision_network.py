# pylint: disable=redefined-outer-name

import logging
import numpy as np
import nengo

# Requires python image library: pip install pillow
from PIL import Image

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from nengo_extras.gui import image_display_function
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Set the nengo logging level to 'info' to display all of the information
# coming back over the ssh connection.
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Set the rng state (using a fixed seed that works)
rng = np.random.RandomState(9)

# Load the MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_train = X_train / 255.0
X_test = X_test / 255.0


# Get information about the image
im_size = int(np.sqrt(X_train.shape[1]))  # Dimension of 1 side of the image
print("Image size: ", im_size)
# Generate the MNIST training and test data
train_targets = to_categorical(y_train)
test_targets = to_categorical(y_test)

# Set up the vision network parameters
n_vis = X_train.shape[1]  # Number of training samples
n_out = train_targets.shape[1]  # Number of output classes
#n_hid = 16000 // (im_size ** 2)  # Number of neurons to use
# Note: the number of neurons to use is limited such that NxD <= 16000,
#       where D = im_size * im_size, and N is the number of neurons to use
n_hid = 500
print("Num hidden neurons: ", n_hid)
gabor_size = (int(im_size / 2.5), int(im_size / 2.5))  # Size of the gabor filt

# Generate the encoders for the neural ensemble
encoders = Gabor().generate(n_hid, gabor_size, rng=rng)
encoders = Mask((im_size, im_size)).populate(encoders, rng=rng, flatten=True)

# Ensemble parameters
max_firing_rates = 100
ens_neuron_type = nengo.neurons.LIF()
ens_intercepts = nengo.dists.Choice([-0.5])
ens_max_rates = nengo.dists.Choice([max_firing_rates])

# Output connection parameters
conn_synapse = 0.1
conn_eval_points = X_train
conn_function = train_targets
conn_solver = nengo.solvers.LstsqL2(reg=0.01)

# Visual input process parameters
presentation_time = 0.25

# Nengo model proper
with nengo.Network(seed=3) as model:
    # Visual input (the MNIST images) to the network
    input_node = nengo.Node(
        nengo.processes.PresentInput(X_test, presentation_time), label="input"
    )
    # Error node
    error = nengo.Node(size_in=n_out, label="error")
    # Output display node
    output_node = nengo.Node(size_in=n_out, label="output class")

    
    ensemble = nengo.Ensemble(n_neurons=n_hid, dimensions=n_vis, neuron_type=ens_neuron_type,
                        encoders=encoders, intercepts=ens_intercepts,
                        max_rates=ens_max_rates, eval_points=conn_eval_points)
                        
    nengo.Connection(input_node, ensemble, synapse=None)
    
    conn = nengo.Connection(ensemble, output_node, function=conn_function, 
                        eval_points=conn_eval_points,
                        learning_rule_type=nengo.PES(0),
                        solver=conn_solver,
                        synapse=conn_synapse)
    
    nengo.Connection(error, conn.learning_rule, synapse=None)
    
    #nengo.Connection(ensemble, error, transform=-1)

    # Projections to and from the ensemble
    #nengo.Connection(post, output_node, synapse=None)

    # Input image display (for nengo_gui)
    image_shape = (1, im_size, im_size)
    display_func = image_display_function(image_shape)
    display_node = nengo.Node(display_func, size_in=input_node.size_out)
    nengo.Connection(input_node, display_node, synapse=None)

    # Output SPA display (for nengo_gui)
    vocab_names = [
        "T-shirt/Top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output_spa = nengo.spa.State(len(vocab_names), subdimensions=n_out, vocab=vocab)
    nengo.Connection(output_node, output_spa.input)
print("Done building")