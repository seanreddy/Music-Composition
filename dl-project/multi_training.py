"""

"""

import os
import numpy as np
import random
import _pickle as pickle
import signal
import tensorflow as tf

from midi_to_statematrix import *
from data import *

batch_width = 10   # number of sequences in a batch
batch_len = 16 * 8 # length of each sequence
division_len = 16  # interval between possible start locations

def load_pieces(dirpath):
    """Load music file (.mid) in a given path
    """
    pieces = {}
    for file in os.listdir(dirpath):
        if file[-4:] not in ('.mid','.MID'):
            continue
        name = file[:-4]
        out_matrix = midi_to_note_statematrix(os.path.join(dirpath, file))
        
        if len(out_matrix) < batch_len:
            continue

        pieces[name] = out_matrix
        print("File {} loaded.".format(name))

    return pieces

def get_piece_segment(pieces):
    """Get segment of pieces
    """
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output) - batch_len, division_len)

    seg_out = piece_output[start:start+batch_len]
    seg_in = note_statematrix_to_inputForm(seg_out)

    return seg_in, seg_out

def get_piece_batch(pieces):
    """
    """
    i, o = zip(*[get_piece_segment(pieces) for _ in range(batch_width)])
    return np.array(i), np.array(o)

def train_piece(model, pieces, epochs, output_name, batch_size=32, start=0, step=319, conservativity=1):
    """Training pieces
    """
    stopflag = [False]
    
    def signal_handler(signame, sf):
        stopflag[0] = True
        
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    
    batch_gen = generate_batch(pieces, batch_size)
    output_file = output_name + ".mid"
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        optimizer = model.optimizer
        loss = model.loss
        
        for i in range(start, start + epochs):
            X_batch, y_batch = next(batch_gen)
            _, error = sess.run((optimizer, loss), feed_dict={
                model.input_layer: X_batch, 
                model.y: y_batch
            })
            
            if i % 100 == 0:
                print("Step {0}, loss = {1}".format(i, error))
            if i % 500 == 0 or (i % 100 == 0 and i < 1000):
                xIpt, xOpt = map(np.array, get_piece_segment(pieces))
            
                # output midi file under output directory
                out_statematrix = sess.run(model.new_song, feed_dict={
                    model.predict_seed: xIpt[0],
                    model.step_to_simulate: [step],
                    model.conservativity: [conservativity]
                })
            # output midi file under output directory
            note_statematrix_to_midi(np.concatenate((np.expand_dims(xOpt[0], 0), out_statematrix)), "output/{0}".format(output_file))
    signal.signal(signal.SIGINT, old_handler)