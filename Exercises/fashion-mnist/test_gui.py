import nengo
import cPickle as pickle


with open("model", "r") as ff:
    model = pickle.load(ff)