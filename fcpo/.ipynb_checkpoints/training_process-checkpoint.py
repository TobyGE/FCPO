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
from replay import ReplayMemory, experience_replay

from environment import Environment

class OrnsteinUhlenbeckNoise:
	''' Noise for Actor predictions. '''

	def __init__(self, action_space_size, mu=0, theta=0.5, sigma=0.2):
		self.action_space_size = action_space_size
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones (self.action_space_size) * self.mu

	def get(self):
		self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.rand (self.action_space_size)
		return self.state


def train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size,
	  discount_factor, nb_episodes, filename_summary, nb_rounds, **env_args):
	''' Algorithm 3 in article. '''

	# Set up summary operators
	def build_summaries():
		episode_reward = tf.Variable (0.)
		tf.summary.scalar ('reward', episode_reward)
		episode_max_Q = tf.Variable (0.)
		tf.summary.scalar ('max_Q_value', episode_max_Q)
		critic_loss = tf.Variable (0.)
		tf.summary.scalar ('critic_loss', critic_loss)

		summary_vars = [episode_reward, episode_max_Q, critic_loss]
		summary_ops = tf.summary.merge_all ()
		return summary_ops, summary_vars

	summary_ops, summary_vars = build_summaries ()
	sess.run (tf.global_variables_initializer ())
	writer = tf.summary.FileWriter (filename_summary, sess.graph)

	# '2: Initialize target network f′ and Q′'
	actor.init_target_network ()
	critic.init_target_network ()

	# '3: Initialize the capacity of replay memory D'
	replay_memory = ReplayMemory(buffer_size)  # Memory D in article
	replay = False

	start_time = time.time ()
	for i_session in range (nb_episodes):  # '4: for session = 1, M do'
		session_reward = 0
		session_Q_value = 0
		session_critic_loss = 0

		# '5: Reset the item space I' is useless because unchanged.
		nb_env = 10
		envs = np.asarray([Environment(**env_args) for i in range(nb_env)])
# 		u = [e.current_user for e in envs]
# 		print(u)
# 		input()
		states = np.array([env.current_state for env in envs])  # '6: Initialize state s_0 from previous sessions'
		
	#         if (i_session + 1) % 10 == 0:  # Update average parameters every 10 episodes
	#             environment.groups = environment.get_groups ()

		exploration_noise = OrnsteinUhlenbeckNoise (history_length * embeddings.size ())

		for t in range (nb_rounds):  # '7: for t = 1, T do'
			# '8: Stage 1: Transition Generating Stage'

			# '9: Select an action a_t = {a_t^1, ..., a_t^K} according to Algorithm 2'
			actions, item_idxes = actor.get_recommendation_list (
				ra_length,
				states.reshape (nb_env, -1),  # TODO + exploration_noise.get().reshape(1, -1),
				embeddings)

			# '10: Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
			for env, state, action, items in zip(envs, states, actions, item_idxes):
				sim_results, rewards, next_state = env.step (action, items)

				# '19: Store transition (s_t, a_t, r_t, s_t+1) in D'
				replay_memory.add (state.reshape (history_length * embeddings.size ()),
								   action.reshape (ra_length * embeddings.size ()),
								   [rewards],
								   next_state.reshape (history_length * embeddings.size ()))

				state = next_state  # '20: Set s_t = s_t+1'

				session_reward += rewards

			# '21: Stage 2: Parameter Updating Stage'
			if replay_memory.size () >= batch_size * nb_env:  # Experience replay
				replay = True
				replay_Q_value, critic_loss = experience_replay (replay_memory, batch_size,
																 actor, critic, embeddings, ra_length,
																 history_length * embeddings.size (),
																 ra_length * embeddings.size (), discount_factor)
				session_Q_value += replay_Q_value
				session_critic_loss += critic_loss

			summary_str = sess.run (summary_ops,
									feed_dict = {summary_vars[0]: session_reward,
												 summary_vars[1]: session_Q_value,
												 summary_vars[2]: session_critic_loss})

			writer.add_summary (summary_str, i_session)

			'''
			print(state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings),
				  state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings, True))
			'''

		str_loss = str ('Loss=%0.4f' % session_critic_loss)
		print (('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (i_session + 1, nb_episodes, session_reward, time.time () - start_time))
		start_time = time.time ()

	writer.close ()
	tf.train.Saver ().save (sess, 'models.h5', write_meta_graph = False)
