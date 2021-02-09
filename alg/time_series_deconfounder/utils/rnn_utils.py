import tensorflow as tf


class AutoregressiveLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, lstm, output_size):
        super(AutoregressiveLSTMCell, self).__init__()

        self.lstm_cell = lstm
        self._output_size = output_size

    @property
    def state_size(self):
        return self.lstm_cell.state_size + self._output_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        actual_states, prev_outputs = tf.split(state,
                                               [self.lstm_cell.state_size, self._output_size],
                                               axis=-1)

        combined_inputs = tf.concat([inputs, prev_outputs], axis=-1)

        with tf.variable_scope("autoregressive"):
            output, state = self.lstm_cell(combined_inputs, actual_states)
            output = tf.layers.dense(output, self._output_size, activation=tf.nn.tanh)

        state = tf.concat([state, output], axis=-1)

        return output, state


def compute_sequence_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)

    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
