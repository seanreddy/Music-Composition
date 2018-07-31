import os
import numpy as np
import tensorflow as tf

from data import *
from midi_to_statematrix import note_statematrix_to_midi

print('Ready to go!')

class Model(object):
    def __init__(self, t_layer_sizes, p_layer_sizes, dropout=0):
        
        # initialization
        tf.reset_default_graph()

        self.t_layer_size = t_layer_sizes
        self.p_layer_sizes = p_layer_sizes
        self.dropout = dropout
      
        # note wise input
        self.t_input_size = 80
        
        # input layer:
        input_layer = tf.placeholder(dtype=tf.float32, shape=(None, None, None, self.t_input_size))
        self.input_layer = input_layer
        
        # output 
        y = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 2))
        self.y = y
                
        # ss commented in the original code, we generated an output for each input 
        # without using the last output as an input  
        input_slice = input_layer[:, :-1]   

        n_batch = tf.shape(input_slice)[0]
        n_time = tf.shape(input_slice)[1]
        n_note = tf.shape(input_slice)[2]
        
        # input for time layers
        time_inputs = tf.reshape(tf.transpose(input_slice, (1, 0, 2, 3)), (n_time, n_batch*n_note, self.t_input_size))
        
        # create LSTM cell for two-layer time model
        t_lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(size),
                                                      output_keep_prob=1-dropout,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32) for size in t_layer_sizes]

        t_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(t_lstm_cells)
        self.t_multi_rnn_cell = t_multi_rnn_cell  

        # create RNN
        time_result, _ = tf.nn.dynamic_rnn(cell = t_multi_rnn_cell,
                                          inputs = time_inputs,
                                          time_major = True,
                                          dtype=tf.float32,
                                          scope='t_rnn')
        
        # input for Note layers
        n_hidden = t_layer_sizes[-1]
        
        time_final = tf.reshape(tf.transpose(tf.reshape(time_result, (n_time, n_batch, n_note, n_hidden)), (2, 1, 0, 3)), (n_note, n_batch*n_time, n_hidden))
        
        start_note_values = tf.zeros([1, n_batch*n_time, 2]) 
        correct_choices = tf.reshape(self.y[:, 1:, :-1, :], (n_note-1, n_batch*n_time, 2)) 
        note_choices_inputs = tf.concat([start_note_values, correct_choices], axis=0) 
        
        note_inputs = tf.concat([time_final, note_choices_inputs], axis=2)  
        
        # create LSTM cell for two-layer pitch model
        n_lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(size),
                                                      output_keep_prob=1-dropout,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32) for size in p_layer_sizes]
        
        n_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(n_lstm_cells)
        self.n_multi_rnn_cell = n_multi_rnn_cell
        
        # two note layers 
        note_result, _ = tf.nn.dynamic_rnn(n_multi_rnn_cell,
                                           note_inputs, 
                                           time_major = True,
                                           dtype=tf.float32) # note_result: output of RNN, _ is state
                
        
        note_final = tf.reshape(tf.layers.dense(tf.reshape(note_result, (n_batch*n_time*n_note, self.p_layer_sizes[-1])),
                                                units=2,
                                                activation=tf.nn.sigmoid,
                                                name='output_layer'), (n_batch, n_time, n_note, 2))
        
        active_notes = input_layer[:, 1:, :, 0:1]
        mask = tf.concat([tf.ones_like(active_notes), active_notes], axis=3)
        
        loglikelihoods = mask * tf.log(2 * note_final*self.y[:,1:] - note_final - self.y[:,1:] + 1)
        self.loss = -tf.reduce_mean(loglikelihoods)
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.setup_predict()
      

   	# setup predict
    def setup_predict(self):
        
        # note axis
        def _step_note(state, indata):
            
            indata = tf.expand_dims(tf.concat((indata, state[1]), axis=-1), axis=0)
            hidden = state[0]

            # note output shape: 50, new_state shape: 100 or 50 
            note_output, new_state = self.n_multi_rnn_cell.call(inputs=indata, state=hidden)
            prob = tf.layers.dense(note_output, units=2, activation=tf.nn.sigmoid, name='output_layer', reuse=True)
            
            shouldplay = tf.cast(tf.random_uniform(shape=(1,)) < (prob[0][0] * self.conservativity), tf.float32)
            shouldartic = shouldplay * tf.cast(tf.random_uniform(shape=(1,)) < prob[0][1], tf.float32)
            output = tf.concat([shouldplay, shouldartic], axis=-1)
            return new_state, output
        
        # time axis
        def _step_time(states, _):
            hidden = states[0] # shape: (notes*t_hidden)*2, e.g. (78*300)*2
            indata = states[1] # shape: notes*note_features, e.g. 78*80
            time = states[2]
            
            # output shape: notes*t_hidden
            output, new_state = self.t_multi_rnn_cell.call(inputs=indata, state=hidden) 
            
            start_note_values = tf.zeros((2))
            
            n_initializer = (self.n_multi_rnn_cell.zero_state(1, tf.float32), start_note_values)
            
            # note_result shape: 78*2
            note_result = tf.scan(_step_note, elems=output, initializer=n_initializer) 
            next_input = map_output_to_input_tf(note_result[1], time)
            next_input = tf.reshape(next_input, (-1,80))
            time += 1
            return new_state, next_input, time, note_result[1] 
        
        # values needed to feed when generating new music
        self.predict_seed = tf.placeholder(dtype=tf.float32, shape=(None,80), name='predict_seed')
        self.step_to_simulate = tf.placeholder(dtype=tf.int32, shape=(1))
        self.conservativity = tf.placeholder(dtype=tf.float32, shape=(1))
        
        num_notes = tf.shape(self.predict_seed)[0]
        
        initializer = (self.t_multi_rnn_cell.zero_state(num_notes, tf.float32), # initial state
                       self.predict_seed,                                       # initial input
                       tf.constant(0),                                          # time
                       tf.placeholder(dtype=tf.float32, shape=(None,2)))        # hold place for note output
        
        elems = tf.zeros(shape=self.step_to_simulate)
        time_result = tf.scan(_step_time, elems=elems, initializer=initializer)
        self.new_music = time_result[3]