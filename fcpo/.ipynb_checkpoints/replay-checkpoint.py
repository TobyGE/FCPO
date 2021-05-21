import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

class ReplayMemory ():
	''' Replay memory D in article. '''

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		# self.buffer = [[row['state'], row['action'], row['reward'], row['n_state']] for _, row in data.iterrows()][-self.buffer_size:] TODO: empty or not?
		self.buffer = []

	def add(self, state, action, reward, n_state):
		self.buffer.append ([state, action, reward, n_state])
		if len (self.buffer) > self.buffer_size:
			self.buffer.pop (0)

	def size(self):
		return len (self.buffer)

	def sample_batch(self, batch_size):
		return random.sample (self.buffer, batch_size)


def experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, state_space_size,
				  action_space_size, discount_factor):
	'''
	  Experience replay.
	  Args:
		replay_memory: replay memory D in article.
		batch_size: sample size.
		actor: Actor network.
		critic: Critic network.
		embeddings: Embeddings object.
		state_space_size: dimension of states.
		action_space_size: dimensions of actions.
	  Returns:
		Best Q-value, loss of Critic network for printing/recording purpose.
	'''

	# '22: Sample minibatch of N transitions (s, a, r, s′) from D'
	samples = replay_memory.sample_batch (batch_size)
	states = np.array ([s[0] for s in samples])
	actions = np.array ([s[1] for s in samples])
	rewards = np.array ([s[2] for s in samples])
	n_states = np.array ([s[3] for s in samples]).reshape (-1, state_space_size)

	# '23: Generate a′ by target Actor network according to Algorithm 2'
	n_actions, _ = actor.get_recommendation_list (ra_length, states, embeddings, target = True)
	n_actions =  n_actions.reshape (-1,action_space_size)

	# Calculate predicted Q′(s′, a′|θ^µ′) value
	target_Q_value = critic.predict_target (n_states, n_actions, [ra_length] * batch_size)

	# '24: Set y = r + γQ′(s′, a′|θ^µ′)'
	expected_rewards = rewards + discount_factor * target_Q_value

	# '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
	critic_Q_value, critic_loss, _ = critic.train (states, actions, [ra_length] * batch_size, expected_rewards)

	# '26: Update the Actor using the sampled policy gradient'
	action_gradients = critic.get_action_gradients (states, n_actions, [ra_length] * batch_size)
	actor.train (states, [ra_length] * batch_size, action_gradients)

	# '27: Update the Critic target networks'
	critic.update_target_network ()

	# '28: Update the Actor target network'
	actor.update_target_network ()

	return np.amax (critic_Q_value), critic_loss