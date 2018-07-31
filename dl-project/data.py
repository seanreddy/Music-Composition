"""

"""

import os
import random
import itertools
import tensorflow as tf

from midi_to_statematrix import *

def start_sentinel():
    
    def note_sentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lower_bound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0] * 66 + [1]

    return [noteSentinel(note) for note in range(upper_bound - lower_bound)]


def get_or_default(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def build_context(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lower_bound) % 12
            context[pitchclass] += 1
    return context


def build_beat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def note_inputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lower_bound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    part_prev_vicinity = list(
        itertools.chain.from_iterable((get_or_default(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]


def generate_batch(pieces, batch_size, piece_length=128):
    while True:
        i,o = zip(*[get_piece_segment(pieces,piece_length) for _ in range(batch_size)])
        yield(i,o)

def get_piece_segment(pieces, piece_length=128, measure_len=16):
    # puece_length means the number of ticks in a training sample, measure_len means number of ticks in a measure
    piece_output = random.choice(list(pieces.values()))
    
    # We just need a segment of a piece as train data, and we want the start of a sample is the start of a measure
    start = random.randrange(0,len(piece_output)-piece_length,measure_len)

    seg_out = piece_output[start:start+piece_length]
    seg_in = note_statematrix_to_inputForm(seg_out)

    return seg_in, seg_out

def map_output_to_input(state, time):
    beat = build_beat(time)

    # Need to convert output state from np.array to list of tuples
    state = state.tolist()
    context = build_context(state)

    # Needs to return a numpy array in order to be wrapped as a tensorflow op
    return np.array([note_inputForm(note, state, context, beat) for note in range(len(state))], dtype=np.float32)

# Tensorflow Graph operations
def map_output_to_input_tf(out_notes, time):
    """Conver output notes to input vectors
    """
    return tf.py_func(map_output_to_input, [out_notes, time], tf.float32)

def note_state_single_to_inputForm(state, time):
    beat = build_beat(time)
    context = build_context(state)
    return [note_inputForm(note, state, context, beat) for note in range(len(state))]

#
# following is key function
#

def note_statematrix_to_inputForm(statematrix):
    inputform = [note_state_single_to_inputForm(state, time) for time, state in enumerate(statematrix)]
    return inputform
