"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import rmsn.libs.net_helpers as helpers

_ACTIVATION_MAP = {'sigmoid': tf.nn.sigmoid,
                   'elu': tf.nn.elu,
                   'tanh': tf.nn.tanh,
                   'linear': lambda x: x}


class StateDumpingRNN(tf.contrib.rnn.RNNCell):
    """ This RNNCell dumps out internal states for lstms"""

    def __init__(self, lstm):

        super(StateDumpingRNN, self).__init__()

        # Check that outputs
        self.lstm_cell = lstm

    @property
    def state_size(self):
        return self.lstm_cell.state_size

    @property
    def output_size(self):
        return self.lstm_cell .state_size

    def call(self, inputs, state):
        output, state = self.lstm_cell(inputs, state)

        return state, state


class Seq2SeqDecoderCell(tf.contrib.rnn.RNNCell):
    """ Decoder cell which allows for feedback, and external inputs during training """
    def __init__(self, lstm, W, b, b_training_mode=False):

        super(Seq2SeqDecoderCell, self).__init__()

        self.lstm_cell = lstm
        self.W = W
        self.b = b
        self._output_size = self.W.get_shape().as_list()[-1]

        self.b_training_mode = b_training_mode

    @property
    def state_size(self):

        if self.b_training_mode:  # use actual inputs
            return self.lstm_cell.state_size
        else:
            return self.lstm_cell.state_size+self._output_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):

        # During training time, we assume that the previous input shape is [batch_size, action_vector + output_vector]
        # Output vectors are assumed to be at the end of the input or state vector (depending on train/test mode respectively)

        if self.b_training_mode:

            actual_states = state
            combined_inputs = inputs

        else:
            actual_states, prev_outputs = tf.split(state,
                                                   [self.lstm_cell.state_size, self._output_size],
                                                   axis=-1)

            combined_inputs = tf.concat([inputs, prev_outputs], axis=-1)


        # TODO: FIX HACK! THis forces this lstm to be in a different scope
        with tf.variable_scope("seq2seq"):
            output, state = self.lstm_cell(combined_inputs, actual_states)
            output = tf.matmul(output, self.W) + self.b

        if not self.b_training_mode:
            state = tf.concat([state, output], axis =-1)

        return output, state


class RnnModel:

    def __init__(self, params):

        # Generic params
        self.net_name = params['net_name']
        self.experiment_name = params['experiment_name']

        # Data params
        self.training_data = params['training_dataset']
        self.validation_data = params['validation_dataset']
        self.test_data = params['test_dataset']
        self.input_size = params['input_size']
        self.output_size = params['output_size']

        # Network params
        self.softmax_size = params['softmax_size']
        self.dropout_rate = params['dropout_rate']
        self.hidden_layer_size = params['hidden_layer_size']
        self.memory_activation_type = params['hidden_activation']
        self.output_activation_type = params['output_activation']
        self.b_use_seq2seq_feedback = params['use_seq2seq_feedback']
        self.b_use_seq2seq_training_mode = params['use_seq2seq_training_mode']

        # Memory Adapter params
        self.b_use_memory_adapter = False if 'use_memory_adapter' not in params else params['use_memory_adapter']
        self.memory_adapter_size = 0 if 'memory_adapter_size' not in params else params['memory_adapter_size']
        self.encoder_state_size = None

        # TODO: FIX THIS HACK FOR LOADING
        # Change scope for seq2seq network - so weights can be loaded later...
        variable_scope_name = "seq2seq" if "seq2seq" in self.net_name else "network"


        with tf.variable_scope(variable_scope_name):
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size,
                                                         activation=_ACTIVATION_MAP[self.memory_activation_type],
                                                         state_is_tuple=False,
                                                         name=variable_scope_name
                                                                if variable_scope_name != "network" else None)
            self.output_activation = _ACTIVATION_MAP[self.output_activation_type]
            self.output_w = tf.get_variable("Output_W",
                                            [self.hidden_layer_size, self.output_size],
                                            dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.output_b = tf.get_variable("Output_b",
                                            [self.output_size],
                                            dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

        # Training params
        self.performance_metric = params['performance_metric']
        self.epochs = params['num_epochs']
        self.minibatch_size = params['minibatch_size']
        self.learning_rate = params['learning_rate']
        self.max_global_norm = params['max_norm']
        self.backprop_length = params['backprop_length']
        self.global_step = tf.get_variable('global_step_tfrnn',
                                           initializer=0,
                                           dtype=np.int32,
                                           trainable=False)

        # Test params
        self.num_prediction_samples = 500

        # Saving params
        self.model_folder = params['model_folder']
        relevant_name_parts = [self.experiment_name,
                                #self.net_name,
                                self.dropout_rate,
                                self.hidden_layer_size,
                                self.epochs,
                                self.minibatch_size,
                                self.learning_rate,
                                self.max_global_norm,
                                self.backprop_length]

        # Check
        if not (self.memory_activation_type == "elu" and self.output_activation_type == "linear"):
            relevant_name_parts += [self.memory_activation_type, self.output_activation_type]

        if self.memory_adapter_size > 0 :
            relevant_name_parts += [self.memory_adapter_size]

        self.serialisation_name = "_".join([str(s) for s in relevant_name_parts])

    def _apply_memory_adapter(self, encoder_states):

        b_single_layer = self.memory_adapter_size == 0  # since externally checked that memory adapter should be applied

        if self.encoder_state_size is None:

            encoder_size = encoder_states.get_shape().as_list()[-1]
            self.encoder_state_size = encoder_size

            if b_single_layer:
                self.memory_adapter_layer = {'W1': tf.get_variable("Adapter_Layer1_W",
                                                                   [self.encoder_state_size,  self.hidden_layer_size*2],
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer()),
                                             'b1': tf.get_variable("Adapter_Layer1_b",
                                                                   [self.hidden_layer_size*2],
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer()),
                                             }
            else:
                self.memory_adapter_layer = {'W1': tf.get_variable("Adapter_Layer1_W",
                                                                   [self.encoder_state_size, self.memory_adapter_size],
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer()),
                                             'b1': tf.get_variable("Adapter_Layer1_b",
                                                                   [self.memory_adapter_size],
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer()),
                                             'W2': tf.get_variable("Adapter_Layer2_W",
                                                                   [self.memory_adapter_size, self.hidden_layer_size*2],
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer()),
                                             'b2': tf.get_variable("Adapter_Layer2_b",
                                                                   [self.hidden_layer_size*2], # LSTM memory is double concated
                                                                   dtype=tf.float32,
                                                                   initializer=tf.contrib.layers.xavier_initializer())
                                             }

        # Use elu and linear to avoid placing any restrictions on the range of internal activations
        memory_activation_fxn = _ACTIVATION_MAP[self.memory_activation_type]
        decoder_states = memory_activation_fxn(tf.matmul(encoder_states, self.memory_adapter_layer['W1'])
                                               + self.memory_adapter_layer['b1'])
        if not b_single_layer:
            decoder_states = memory_activation_fxn(tf.matmul(decoder_states, self.memory_adapter_layer['W2'])
                                                   + self.memory_adapter_layer['b2'])

        return decoder_states


    def get_prediction_graph(self, use_validation_set,
                             with_dropout=True,
                             placeholder_time_steps=None,
                             b_use_state_initialisation=False,
                             b_dump_all_states=False):

        if placeholder_time_steps:
            data_chunk = {}
            data_chunk['inputs'] = tf.placeholder(tf.float32,[None, placeholder_time_steps, self.input_size])
            data_chunk['sequence_lengths'] = tf.placeholder(tf.float32,[None])  # Length
        else:
            if use_validation_set is None:
                dataset = self.training_data.batch(self.minibatch_size)
            elif use_validation_set:
                dataset = self.validation_data.batch(self.minibatch_size)
            else:
                dataset = self.test_data.batch(self.minibatch_size)

            iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                       dataset.output_shapes)

            initializer = iterator.make_initializer(dataset)
            data_chunk = iterator.get_next()

        if b_use_state_initialisation:
            if 'initial_states' not in data_chunk:
                raise ValueError("State initialisations not present!")

            initial_states = tf.cast(data_chunk['initial_states'], tf.float32)
        else:
            initial_states = None

        output = self._build_prediction_graph(data_chunk,
                                              with_dropout=with_dropout,
                                              initial_states=initial_states,
                                              b_dump_all_states=b_dump_all_states)

        if placeholder_time_steps:
            output['input_holder'] = data_chunk['inputs']
            output['sequence_length_holder'] = data_chunk['sequence_lengths']
        else:
            output['initializer'] = initializer

        return output

    def _build_prediction_graph(self, data_chunk, with_dropout=True, initial_states=None,
                                b_dump_all_states=False):

        # output_minibatch = tf.cast(data_chunk['outputs'], tf.float32)
        # active_entries = tf.cast(data_chunk['active_entries'], tf.float32)
        input_minibatch = tf.cast(data_chunk['inputs'], tf.float32)
        sequence_lengths = tf.cast(data_chunk['sequence_lengths'], tf.int32)

        time_steps = input_minibatch.get_shape().as_list()[1]

        # Setup graph now
        outputs = []
        states_list = []
        if with_dropout:
            num_samples = self.num_prediction_samples
            keep_probs = (1 - self.dropout_rate)
        else:
            num_samples = 1
            keep_probs = 1.0

        lstm_additional_size = self.output_size \
            if not self.b_use_seq2seq_training_mode and self.b_use_seq2seq_feedback \
            else 0
        cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell,
                                             input_keep_prob=keep_probs,
                                             output_keep_prob=keep_probs,
                                             state_keep_prob=keep_probs,
                                             variational_recurrent=True,
                                             input_size=input_minibatch.shape[2] + lstm_additional_size,
                                             dtype=tf.float32)

        # Extension for feedback loops in seq2seq architecture
        if self.b_use_seq2seq_feedback:
            cell = Seq2SeqDecoderCell(cell, self.output_w, self.output_b, b_training_mode=False)

        # Extension for memory adapter
        if self.b_use_memory_adapter:

            if initial_states is None:
                raise ValueError("Memory adapter requires initial states!")

            initial_states = self._apply_memory_adapter(initial_states)


        for i in range(num_samples):

            val, states = tf.nn.dynamic_rnn(cell,
                                            input_minibatch,
                                            initial_state=initial_states,  # None for default
                                            dtype=tf.float32,
                                            sequence_length=sequence_lengths)

            if b_dump_all_states:
                state_dumping_cell = StateDumpingRNN(cell)
                all_states, dumped_states = tf.nn.dynamic_rnn(state_dumping_cell,
                                                              input_minibatch,
                                                              initial_state=initial_states,  # None for default
                                                              dtype=tf.float32,
                                                              sequence_length=sequence_lengths)
            else:
                all_states = states  # just dump one state - used to speed up training while enforcing function params

            # Linear output layer
            flattened_val = tf.reshape(val, [-1, self.hidden_layer_size])

            if self.b_use_seq2seq_feedback:
                logits = flattened_val
            else:
                logits = tf.matmul(flattened_val, self.output_w) + self.output_b

            if self.softmax_size != 0:

                logits = tf.reshape(logits, [-1, time_steps, self.output_size])

                core_outputs, softmax_outputs = tf.split(logits,
                                                         [self.output_size - self.softmax_size, self.softmax_size],
                                                         axis=2)

                output = tf.concat([self.output_activation(core_outputs), tf.nn.softmax(softmax_outputs, axis=2)],
                                   axis=2)

            else:

                output = self.output_activation(logits)
                output = tf.reshape(output, [-1, time_steps, self.output_size])

            outputs.append(tf.expand_dims(output, 0))
            states_list.append(tf.expand_dims(all_states, 0))

        # Dumping output
        samples = tf.concat(outputs, axis=0)
        mean_estimate = tf.reduce_mean(samples, axis=0)
        upper_bound = tf.contrib.distributions.percentile(samples, q=95.0, axis=0)
        lower_bound = tf.contrib.distributions.percentile(samples, q=5.0, axis=0)

        # Averages across all samples - no difference for single sample
        ave_state = tf.reduce_mean(tf.concat(states_list, axis=0), axis=0)

        return {'mean': mean_estimate, 'upper_bound': upper_bound, 'lower_bound': lower_bound, 'ave_states': ave_state}

    def get_training_graph(self,
                           use_truncated_bptt=True,
                           b_stub_front=True,
                           b_use_state_initialisation=True):

        training_dataset = self.training_data.shuffle(buffer_size=10000)  \
                            .batch(self.minibatch_size) \
                            .repeat(self.epochs)

        iterator = training_dataset.make_one_shot_iterator()
        data_chunk = iterator.get_next()

        input_minibatch = tf.cast(data_chunk['inputs'], tf.float32)
        output_minibatch = tf.cast(data_chunk['outputs'], tf.float32)
        active_entries = tf.cast(data_chunk['active_entries'], tf.float32)
        sequence_lengths = tf.cast(data_chunk['sequence_lengths'], tf.int32)

        if b_use_state_initialisation:
            if 'initial_states' not in data_chunk:
                raise ValueError("State initialisations not present!")

            initial_states = tf.cast(data_chunk['initial_states'], tf.float32)

            # Extension for memory adapter
            if self.b_use_memory_adapter:
                if initial_states is None:
                    raise ValueError("Memory adapter requires initial states!")
                initial_states = self._apply_memory_adapter(initial_states)

        else:
            initial_states = None

        if 'propensity_weights' in data_chunk:
            weights = tf.cast(data_chunk['propensity_weights'], tf.float32)
        else:
            weights = 1

        keep_probs = (1 - self.dropout_rate)

        # Setup graph now
        lstm_additional_size = self.output_size \
                                if not self.b_use_seq2seq_training_mode and self.b_use_seq2seq_feedback \
                                else 0
        cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell,
                                             input_keep_prob=keep_probs,
                                             output_keep_prob=keep_probs,
                                             state_keep_prob=keep_probs,
                                             variational_recurrent=True,
                                             input_size=input_minibatch.shape[2] + lstm_additional_size,
                                             dtype=tf.float32)

        if self.b_use_seq2seq_feedback:
            cell = Seq2SeqDecoderCell(cell, self.output_w, self.output_b,
                                      b_training_mode=self.b_use_seq2seq_training_mode)

        # Stack up the dynamic RNNs for T-BPTT.

        # Splitting it up
        total_timesteps = input_minibatch.get_shape().as_list()[1]
        num_slices = int(total_timesteps/self.backprop_length)
        chunk_sizes = [self.backprop_length for i in range(num_slices)]
        odd_size = total_timesteps - self.backprop_length*num_slices

        # get all the chunks
        if odd_size > 0:
            if b_stub_front:
                chunk_sizes = [odd_size] + chunk_sizes
            else:
                chunk_sizes = chunk_sizes + [odd_size]

        # Implement TF style Truncated-backprop through time
        outputs = []
        start = 0
        states = initial_states
        for chunk_size in chunk_sizes:

            input_chunk = tf.slice(input_minibatch, [0, start, 0], [-1, chunk_size, self.input_size])
            if states is not None and use_truncated_bptt:
                val, states = tf.nn.dynamic_rnn(cell,
                                                input_chunk,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32,
                                                initial_state=states)
            else:
                val, states = tf.nn.dynamic_rnn(cell,
                                                input_chunk,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32)

            # Linear output layer
            flattened_val = tf.reshape(val, [-1, self.hidden_layer_size])
            if self.b_use_seq2seq_feedback:
                logits = flattened_val
            else:
                logits = tf.matmul(flattened_val, self.output_w) + self.output_b

            if self.softmax_size !=0:

                logits = tf.reshape(logits, [-1, chunk_size, self.output_size])

                core_outputs, softmax_outputs = tf.split(logits,
                                                         [self.output_size - self.softmax_size, self.softmax_size],
                                                         axis=2)

                output = tf.concat([self.output_activation(core_outputs), tf.nn.softmax(softmax_outputs, axis=2)], axis=2)

            else:
                output = self.output_activation(logits)
                output = tf.reshape(output, [-1, chunk_size, self.output_size])

            outputs.append(output)

            # break links between states for truncated bptt
            states = tf.identity(states)

            # Starting point
            start += chunk_size

        # Dumping output
        predictions = tf.concat(outputs, axis=1)

        # Split out the softmax components
        if self.softmax_size > 0:

            original_vs_softmax_size = [self.output_size -self.softmax_size, self.softmax_size]
            predictions, softmax_predictions = tf.split(predictions, original_vs_softmax_size, axis=2)
            output_minibatch, softmax_output_minibatch = tf.split(output_minibatch, original_vs_softmax_size, axis=2)
            active_entries, softmax_active = tf.split(active_entries, original_vs_softmax_size, axis=2)

        # Compute loss function
        if self.performance_metric == "mse":
            loss = tf.reduce_sum(tf.square(predictions - output_minibatch) * active_entries * weights) \
                   / tf.reduce_sum(active_entries)  # cos some zero entires

        elif self.performance_metric == "xentropy":
            loss = tf.reduce_sum((output_minibatch * -tf.log(predictions + 1e-8)
                                  + (1 - output_minibatch) * -tf.log(1 - predictions + 1e-8))
                                 * active_entries * weights) \
                   / tf.reduce_sum(active_entries)
        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))

        if self.softmax_size > 0:
            loss += tf.reduce_sum(softmax_output_minibatch * -tf.log(softmax_predictions + 1e-8)
                                  * softmax_active * weights) \
                   / tf.reduce_sum(softmax_active)

        optimiser = helpers.get_optimization_graph(loss,
                                                   learning_rate=self.learning_rate,
                                                   max_global_norm=self.max_global_norm,
                                                   global_step=self.global_step)
        # Parcel outputs
        handles = {'loss': loss,
                  'optimiser': optimiser}

        return handles



