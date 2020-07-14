from collections import deque
import numpy as np


class UniformReplay:
    """
    Class for the uniform experience replay
    """

    def __init__(self, maxlen):
        self.experiences = deque(maxlen=maxlen)

    def sample(self, batch_size):
        ids = np.random.randint(len(self.experiences), size=batch_size)
        states = []
        actions = []
        rewards = []
        states_next = []
        dones = []
        for idx in ids:
            states.append(self.experiences[idx][0])
            actions.append(self.experiences[idx][1])
            rewards.append(self.experiences[idx][2])
            states_next.append(self.experiences[idx][3])
            dones.append(self.experiences[idx][4])
        return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(states_next), np.asarray(dones)

    def add(self, experience):
        self.experiences.append(experience)


class PrioritizedReplay:

    def __init__(self, maxlen, epsilon=0.001):
        self.experiences = deque(maxlen=maxlen)
        # Parameters initialised according to the paper values
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.01
        self.epsilon = epsilon

    def add(self, experience):
        # Adding the experience and appending a td_error of 1 as initialisation
        self.experiences.append(experience + (1,))

    def sample(self, batch_size):
        # Updating beta value
        self.beta = np.min([1., self.beta + self.beta_increment])

        # Calculate the value of each element
        probabilities = np.asarray(list(map(lambda x: x[5] ** self.alpha, self.experiences)))
        # Normalize it in order to have probabilities
        probabilities = probabilities / np.sum(probabilities)
        # Pick batch_size indexes according to the given probability distribution
        ids = np.random.choice(len(self.experiences), batch_size, p=probabilities)
        states = []
        actions = []
        rewards = []
        states_next = []
        dones = []

        # Calculate the importance sampling value
        importance_samplings = np.power(len(self.experiences) * probabilities[ids], -self.beta)
        importance_samplings /= importance_samplings.max()

        # Build the data structures that have to be returned
        for idx in ids:
            states.append(self.experiences[idx][0])
            actions.append(self.experiences[idx][1])
            rewards.append(self.experiences[idx][2])
            states_next.append(self.experiences[idx][3])
            dones.append(self.experiences[idx][4])

        return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(states_next), np.asarray(
            dones), ids, importance_samplings

    # Method to update the priority of the samples after have been fed to the network
    def update_priority(self, ids, td_errors):

        for td_index, idx in enumerate(ids):
            td_errors = np.abs(td_errors) + self.epsilon

            # Update experiences td_error
            self.experiences[idx] = (
                self.experiences[idx][0], self.experiences[idx][1], self.experiences[idx][2], self.experiences[idx][3],
                self.experiences[idx][4],
                td_errors[td_index])
