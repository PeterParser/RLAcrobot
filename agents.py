import tensorflow as tf
import numpy as np


class DQNAgent:

    def __init__(self, hidden_layers, state_spec, action_spec, buffer, learning_rate, is_prioritized, is_double):
        self.buffer = buffer
        self.training_network = tf.keras.models.Sequential()
        self.training_network.add(tf.keras.layers.InputLayer(input_shape=(state_spec,)))

        for hidden_layer in hidden_layers:
            self.training_network.add(tf.keras.layers.Dense(hidden_layer, activation='relu'))

        self.training_network.add(tf.keras.layers.Dense(action_spec, activation='linear'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.is_prioritized = is_prioritized
        self.is_double = is_double
        self.loss = tf.keras.losses.mean_squared_error

    def play_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.training_network.output_shape[1])
        else:
            return np.argmax(self.training_network.predict(np.atleast_2d(state))[0])

    def train(self, gamma, batch_size):
        if self.is_prioritized:
            states, actions, rewards, states_next, dones, ids, weight = self.buffer.sample(batch_size)

        else:
            states, actions, rewards, states_next, dones = self.buffer.sample(batch_size)

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


class ActorCriticAgent:

    def __init__(self, hidden_layers_actor, hidden_layers_critic, state_spec, action_spec, learning_rate_actor,
                 learning_rate_critic):
        self.actor_network = tf.keras.models.Sequential()
        self.critic_network = tf.keras.models.Sequential()

        # Input layers
        self.actor_network.add(tf.keras.layers.InputLayer(input_shape=(state_spec,)))
        self.critic_network.add(tf.keras.layers.InputLayer(input_shape=(state_spec,)))

        # Hidden layers
        for hidden_layer in hidden_layers_actor:
            self.actor_network.add(tf.keras.layers.Dense(hidden_layer, activation='relu'))
        for hidden_layer in hidden_layers_critic:
            self.critic_network.add(tf.keras.layers.Dense(hidden_layer, activation='relu'))

        # Output layers
        self.actor_network.add(tf.keras.layers.Dense(action_spec, activation='softmax', dtype='float64'))
        self.critic_network.add(tf.keras.layers.Dense(1, activation='linear', dtype='float64'))

        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate_critic)
        self.loss = tf.keras.losses.mean_squared_error

    def play_action(self, state):
        probabilities = self.actor_network(np.atleast_2d(state))
        selection_probabilities = probabilities[0] / np.sum(probabilities[0])
        action = np.random.choice(self.actor_network.output_shape[1], p=selection_probabilities)
        return action

    def play_and_train(self, state, env, gamma):
        with tf.GradientTape(persistent=True) as tape:
            probabilities = self.actor_network(np.atleast_2d(state))
            # I need to normalize probability because numpy wants that the sum must be 1 and the softmax gives me
            # 0.99999999

            # Stop recording needed because predict method doesn't work inside gradient tape
            # And for not taking the gradient of target in order to do a semigradient update
            with tape.stop_recording():
                selection_probabilities = probabilities[0] / np.sum(probabilities[0])
                action = np.random.choice(self.actor_network.output_shape[1], p=selection_probabilities)
                next_state, reward, done, _ = env.step(action)
                target = reward + (1 - done) * self.critic_network.predict(np.atleast_2d(next_state)) * gamma

            predicted = self.critic_network(np.atleast_2d(state))
            critic_loss = self.loss(target, predicted)
            # Need to use tf.math.log instead of np.log because the gradient tape cannot track np.log
            actor_loss = -tf.math.log(probabilities[0, action]) * (target - predicted)

        actor_gradient = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        critic_gradient = tape.gradient(critic_loss, self.critic_network.trainable_weights)
        self.optimizer_actor.apply_gradients(zip(actor_gradient, self.actor_network.trainable_weights))
        self.optimizer_critic.apply_gradients(zip(critic_gradient, self.critic_network.trainable_weights))

        return next_state, reward, done
