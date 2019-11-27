# pylint: disable=redefined-outer-name

import logging
import numpy as np
import nengo

# Requires python image library: pip install pillow
from PIL import Image

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from nengo_extras.gui import image_display_function


# ------ MISC HELPER FUNCTIONS -----
def resize_img(img, im_size, im_size_new):
    # Resizes the MNIST images to a smaller size so that they can be processed
    # by the FPGA (the FPGA currently has a limitation on the number of
    # dimensions and neurons that can be built into the network)
    # Note: Requires the python PIL (pillow) library to work
    img = Image.fromarray(img.reshape((im_size, im_size)) * 256, "F")
    img = img.resize((im_size_new, im_size_new), Image.ANTIALIAS)
    return np.array(img.getdata(), np.float32) / 256.0


def one_hot(labels, c=None):
    # One-hot function. Converts a given class and label list into a vector
    # of 0's (no class match) and 1's (class match)
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y


# ---------------- BOARD SELECT ----------------------- #
# Change this to your desired device name
board = "de1"
# ---------------- BOARD SELECT ----------------------- #

# Set the nengo logging level to 'info' to display all of the information
# coming back over the ssh connection.
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Set the rng state (using a fixed seed that works)
rng = np.random.RandomState(9)

# Load the MNIST data
(X_train, y_train), (X_test, y_test) = load_mnist()


X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

# Get information about the image
im_size = int(np.sqrt(X_train.shape[1]))  # Dimension of 1 side of the image

# Generate the MNIST training and test data
train_targets = one_hot(y_train, 10)
test_targets = one_hot(y_test, 10)

# Set up the vision network parameters
n_vis = X_train.shape[1]  # Number of training samples
n_out = train_targets.shape[1]  # Number of output classes
n_hid = 16000 // (im_size ** 2)  # Number of neurons to use
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
                        learning_rule_type=nengo.PES(learning_rate=0),
                        solver=conn_solver,
                        synapse=conn_synapse)
    
    nengo.Connection(error, conn.learning_rule, synapse=None)

    # Projections to and from the ensemble
    #nengo.Connection(post, output_node, synapse=None)

    # Input image display (for nengo_gui)
    image_shape = (1, im_size, im_size)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=input_node.size_out)
    nengo.Connection(input_node, display_node, synapse=None)

    # Output SPA display (for nengo_gui)
    vocab_names = [
        "ZERO",
        "ONE",
        "TWO",
        "THREE",
        "FOUR",
        "FIVE",
        "SIX",
        "SEVEN",
        "EIGHT",
        "NINE",
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