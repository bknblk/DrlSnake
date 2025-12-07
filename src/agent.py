import numpy as np
import random
from src.network import create_dqn
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape=(10,10,10), n_actions = 6):
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.policy_net = create_dqn(state_shape, n_actions)
        self.target_net = create_dqn(state_shape, n_actions)
        self.target_net.set_weights(self.policy_net.get_weights())

        self.memory = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon = None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            state_input = state[np.newaxis, ... , np.newaxis]
            q_values = self.policy_net.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
            return action


    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        current_q = self.policy_net.predict(states, verbose=0)
        next_q = self.target_net.predict(next_states, verbose=0)
        target_q = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        history = self.policy_net.fit(states, target_q, epochs=1, verbose=0)
        return history.history['loss'][0]

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)






