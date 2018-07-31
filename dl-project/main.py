import numpy as np
import model
import data
import multi_training
import _pickle as pickle
from midi_to_statematrix import *

if __name__ == '__main__':
    
	# load midi files from music directory
    print("Loading data . . .")
    pieces = multi_training.load_pieces("music")

    # Building model
    biaxial_model = model.Model(t_layer_sizes=[300,300], p_layer_sizes=[100,50])
    
    print("Start training . . .")
    multi_training.train_piece(biaxial_model, pieces, epochs=2000, output_name="composition_1216_03")
    print("Training complete . . .")