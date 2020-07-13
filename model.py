import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, hidden_layers, state_spec, action_spec, buffer, learning_rate, is_prioritized, is_double):
        self.buffer = buffer
        self.training_network = tf.keras.models.Sequential()
        self.training_network.add(tf.keras.layers.InputLayer(input_shape=(state_spec,)))

        if is_double:
            self.target_network = tf.keras.models.Sequential()
            self.target_network.add(tf.keras.layers.InputLayer(input_shape=(state_spec,)))

        for hidden_layer in hidden_layers:
            self.training_network.add(tf.keras.layers.Dense(hidden_layer, activation='relu'))
            if is_double:
                self.target_network.add(tf.keras.layers.Dense(hidden_layer, activation='relu'))

        self.training_network.add(tf.keras.layers.Dense(action_spec, activation='linear'))

        if is_double:
            self.target_network.add(tf.keras.layers.Dense(action_spec, activation='linear'))

        # Make them to have the same weights at start
        if is_double:
            self.target_network.set_weights(self.training_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.is_prioritized = is_prioritized
        self.is_double = is_double
        self.loss = tf.keras.losses.mean_squared_error

    def play_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.training_network.output_shape[1])
        else:
            return np.argmax(self.training_network.predict(np.atleast_2d(states))[0])

    def train(self, gamma, batch_size):
        if self.is_prioritized:
            states, actions, rewards, states_next, dones, ids, weight = self.buffer.sample(batch_size)

        else:
            states, actions, rewards, states_next, dones = self.buffer.sample(batch_size)

        if self.is_double:
            q_value_next_time_step_training = self.target_network(states_next)
            q_value_next_time_step_target = self.training_network(states_next)
            action_indexes = np.argmax(q_value_next_time_step_training, axis=1)
            q_value_next_time_step_target = np.array(q_value_next_time_step_target)
            q_values_next_time_step = q_value_next_time_step_target[:, action_indexes][:, 0]

            target = rewards + (1 - dones) * gamma * q_values_next_time_step

        else:
            target = rewards + (1 - dones) * gamma * np.max(self.training_network.predict(states_next), axis=1)

        mask = tf.one_hot(actions, self.training_network.output_shape[1])
        with tf.GradientTape() as tape:
            q_values = self.training_network(states)
            predicted = tf.reduce_sum(q_values * mask, axis=1)

            if self.is_prioritized:
                loss = tf.reduce_mean(weight * self.loss(target, predicted))
            else:
                loss = tf.reduce_mean(self.loss(target, predicted))

        if self.is_prioritized:
            self.buffer.update_priority(ids, (target - predicted))

        variables = self.training_network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def update_target(self):
        self.target_network.set_weights(self.training_network.get_weights())
