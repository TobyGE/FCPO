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

class Actor ():
	''' Policy function approximator. '''
	def __init__(self, sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embedding_size,
				 tau, learning_rate, scope='actor'):
		self.sess = sess
		self.state_space_size = state_space_size
		self.action_space_size = action_space_size
		self.batch_size = batch_size
		self.ra_length = ra_length
		self.history_length = history_length
		self.embedding_size = embedding_size
		self.tau = tau
		self.learning_rate = learning_rate
		self.scope = scope

		with tf.compat.v1.variable_scope (self.scope):
			# Build Actor network
			self.action_weights, self.state, self.sequence_length = self._build_net ('estimator_actor')
			self.network_params = tf.trainable_variables ()

			# Build target Actor network
			self.target_action_weights, self.target_state, self.target_sequence_length = self._build_net (
				'target_actor')
			self.target_network_params = tf.trainable_variables ()[len (
				self.network_params):]  # TODO: why sublist [len(x):]? Maybe because its equal to network_params + target_network_params

			# Initialize target network weights with network weights (θ^π′ ← θ^π)
			self.init_target_network_params = [self.target_network_params[i].assign (self.network_params[i])
											   for i in range (len (self.target_network_params))]

			# Update target network weights (θ^π′ ← τθ^π + (1 − τ)θ^π′)
			self.update_target_network_params = [self.target_network_params[i].assign (
				tf.multiply (self.tau, self.network_params[i]) +
				tf.multiply (1 - self.tau, self.target_network_params[i]))
				for i in range (len (self.target_network_params))]

			# Gradient computation from Critic's action_gradients
			self.action_gradients = tf.placeholder (tf.float32, [None, self.action_space_size])
			gradients = tf.gradients (
				tf.reshape (self.action_weights, [self.batch_size, self.action_space_size], name = '42222222222'),
				self.network_params,
				self.action_gradients)
			params_gradients = list (map (lambda x: tf.div (x, self.batch_size * self.action_space_size), gradients))

			# Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s)
			self.optimizer = tf.train.AdamOptimizer (self.learning_rate).apply_gradients (
				zip (params_gradients, self.network_params))


	def _build_net(self, scope):
		''' Build the (target) Actor network. '''
		def gather_last_output(data, seq_lens):
			def cli_value(x, v):
				y = tf.constant (v, shape = x.get_shape (), dtype = tf.int64)
				x = tf.cast (x, tf.int64)
				return tf.where (tf.greater (x, y), x, y)

			batch_range = tf.range (tf.cast (tf.shape (data)[0], dtype = tf.int64), dtype = tf.int64)
			tmp_end = tf.map_fn (lambda x: cli_value (x, 0), seq_lens - 1, dtype = tf.int64)
			indices = tf.stack ([batch_range, tmp_end], axis = 1)
			return tf.gather_nd (data, indices)

		with tf.variable_scope (scope):
			# Inputs: current state, sequence_length
			# Outputs: action weights to compute the score Equation (6)
			state = tf.compat.v1.placeholder (tf.float32, [None, self.state_space_size], 'state')
			state_ = tf.reshape (state, [-1, self.history_length, self.embedding_size])
			sequence_length = tf.placeholder (tf.int32, [None], 'sequence_length')
			cell = tf.nn.rnn_cell.GRUCell (self.embedding_size,
											activation = tf.nn.relu,
											kernel_initializer = tf.initializers.random_normal (),
											bias_initializer = tf.zeros_initializer ())
			outputs, _ = tf.nn.dynamic_rnn (cell, state_, dtype = tf.float32, sequence_length = sequence_length)
			last_output = gather_last_output (outputs, sequence_length)  # TODO: replace by h
			x = tf.keras.layers.Dense (self.ra_length * self.embedding_size) (last_output)
			action_weights = tf.reshape (x, [-1, self.ra_length, self.embedding_size])

		return action_weights, state, sequence_length

	def train(self, state, sequence_length, action_gradients):
		'''  Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s). '''
		self.sess.run (self.optimizer,
					   feed_dict = {
						   self.state: state,
						   self.sequence_length: sequence_length,
						   self.action_gradients: action_gradients})

	def predict(self, state, sequence_length):
		return self.sess.run (self.action_weights,
							  feed_dict = {
								  self.state: state,
								  self.sequence_length: sequence_length})

	def predict_target(self, state, sequence_length):
		return self.sess.run (self.target_action_weights,
							  feed_dict = {
								  self.target_state: state,
								  self.target_sequence_length: sequence_length})

	def init_target_network(self):
		self.sess.run (self.init_target_network_params)

	def update_target_network(self):
		self.sess.run (self.update_target_network_params)

	def get_recommendation_list(self, ra_length, noisy_state, embeddings, target=False):
		'''
		Algorithm 2
		Args:
		  ra_length: length of the recommendation list.
		  noisy_state: current/remembered environment state with noise.
		  embeddings: Embeddings object.
		  target: boolean to use Actor's network or target network.
		Returns:
		  Recommendation List: list of embedded items as future actions.
		'''

		batch_size = noisy_state.shape[0]


		# '1: Generate w_t = {w_t^1, ..., w_t^K} according to Equation (5)'
		method = self.predict_target if target else self.predict
		
		
		weights = tf.cast(method (noisy_state, [ra_length] * batch_size), tf.float64)
		scores = tf.linalg.matmul(weights, tf.transpose(embeddings.get_embedding_vector()))
		item_idxes = tf.math.argmax(scores,2)
# 		print(weights.shape)
# 		print(scores.shape)
# 		print(item_idxes.shape)
# 		input()
		with tf.Session() as sess:
			item_idxes = sess.run(item_idxes)
		actions = np.array([embeddings.embed(i) for i in item_idxes])
# 		print(actions.shape)
# 		input()
		
		
		# '3: Score items in I according to Equation (6)'
# 		weights = method (noisy_state, [ra_length] * batch_size)
# 		scores = np.array ([[[get_score (weights[i][k], embedding, batch_size)
# 							  for embedding in embeddings.get_embedding_vector ()]
# 							 for k in range (ra_length)]
# 							for i in range (batch_size)])

# 		# '8: return a_t'
# 		actions = np.array ([[embeddings.get_embedding (np.argmax (scores[i][k]))
# 						   for k in range (ra_length)]
# 						  for i in range (batch_size)])
# 		item_idxes = np.array ([[np.argmax(scores[i][k])
# 						   for k in range (ra_length)]
# 						  for i in range (batch_size)])
# 		print(item_idxes)
# 		input()
		return actions, item_idxes
