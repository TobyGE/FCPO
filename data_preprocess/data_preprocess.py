import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
import copy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class DataPreprocessor():
	def __init__(self, datapath, itempath):
		'''
		Load data from the DB MovieLens
		List the users and the items
		List all the users historic
		'''
		self.data  = self.load_data(datapath, itempath)
		userId = np.array(self.data['userId'].values.tolist())
		itemId = np.array(self.data['itemId'].values.tolist())
		self.data['userId'] = list(userId)
		self.data['itemId'] = list(itemId)
		
		self.users = self.data['userId'].unique()   #list of all users
		self.items = self.data['itemId'].unique()   #list of all items
		self.nb_user = len(self.users)
		self.nb_item = len(self.items)
		print('total num of users:',self.nb_user)
		print('total num of items:',self.nb_item)
		#a list contains the rating history of each user
		self.histo = self.gen_histo()


	def load_data(self, dataname, datapath):
		'''
		Load the data and merge the name of each movie.
		A row corresponds to a rate given by a user to a movie.

		 Parameters
		----------
		datapath :  string
					path to the data 100k MovieLens
					contains usersId;itemId;rating
		itempath :  string
					path to the data 100k MovieLens
					contains itemId;itemName
		 Returns
		-------
		result :    DataFrame
					Contains all the ratings
		'''
# 		if dataname == 'ml-100k':
# 			data = pd.read_csv(datapath, sep=',',
# 						names=['userId', 'itemId', 'rating', 'timestamp'],
# 						dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32})
# 		if dataname == 'ml-1m':
# 			data = pd.read_csv(datapath, sep=',',
# 						names=['userId', 'itemId', 'rating', 'timestamp'],
# 						dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32})
# 		if dataname == 'ml-20m':
# 			data = pd.read_csv(datapath,
# 						names=['userId', 'itemId', 'rating', 'timestamp'],
# 						dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32})
# 		if dataname == 'cd':
# 			data = pd.read_csv(datapath, sep=',',
# 						names=['userId', 'itemId', 'rating', 'timestamp'],
# 						dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32},
# 						engine='python')
# 		if dataname == 'ciao' or dataname == 'epinions':
# 			data = pd.read_csv(datapath, sep=',',
# 						names=['userId', 'itemId', 'category','rating', 'helpfulness','timestamp'],
# 						dtype={'userId':np.int32,'category':np.int32,'itemId':np.int32,'rating':np.float64,'helpfulness':np.int32,'timestamp':np.int32},
# 						engine='python')    
		data = pd.read_csv(datapath, sep=',',
						names=['userId', 'itemId', 'rating', 'timestamp'],
						dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32})
		return data


	def gen_histo(self):
		'''
		Group all rates given by users and store them from older to most recent.

		Returns
		-------
		result :    List(DataFrame)
					List of the historic for each user
		'''
		print('start generating user history...')
		historic_users = []
		for i, u in tqdm(enumerate(self.users)):
			temp = self.data[self.data['userId'] == u]
			temp = temp.sort_values ('timestamp').reset_index ()
			temp.drop ('index', axis = 1, inplace = True)
			historic_users.append (temp)
		return historic_users

	
	def sample_histo_v6(self, user_histo, pivot_rating, nb_states, nb_actions):
		n = len(user_histo)

		states = []
		actions = []
		rewards = []
		states_prime = []
		done = []
		
		state_len = nb_states
		action_len = nb_actions

		item_list = user_histo['itemId'].values.tolist()
		click_list = user_histo['rating'].values.tolist()

		initial_state = []
		initial_end = 0
		for i in range(len(item_list)):
			if click_list[i] > pivot_rating and len(initial_state) < state_len:
				initial_state.append(item_list[i])
				initial_end = i

		if len(initial_state) == state_len and (initial_end + action_len <= len(item_list)):
			current_state = copy.copy(initial_state)
			for i in range(initial_end+1,len(item_list),action_len):
				if i+action_len <= len(item_list):
					actions.append(item_list[i:i+action_len])
					rewards.append(click_list[i:i+action_len])
					states.append(copy.copy(current_state))
					done.append(False)
					
					for j in range(i,i+action_len):
						if click_list[j] > pivot_rating:
							current_state.append(item_list[j])
							del current_state[0]
					states_prime.append(copy.copy(current_state))
		if len(done) > 0:
			done[-1] = True
		return states, actions, rewards, states_prime, done

	def sample_histo_v5(self, user_histo, nb_states, pivot_rating):
		prop_histo = user_histo[user_histo['rating'] >= pivot_rating]
		if len(prop_histo) > nb_states:
			user = user_histo['userId'][0]
			initial_state =  prop_histo[0:nb_states]['itemId'].values.tolist()
			initial_rewards = prop_histo[0:nb_states]['rating'].values.tolist()
			user_history =  prop_histo[nb_states:]['itemId'].values.tolist()
			rewards = prop_histo[nb_states:]['rating'].values.tolist()
		return user, initial_state, initial_rewards, user_history, rewards
	
	
	def write_csv(self, train_filename=None, test_filename=None, train_test_ratio=0.8, pivot_rating=0, nb_states=5, nb_actions=1):
		users = []
		initial_states = []
		initial_rewards = []
		user_histories = []
		rewards = []
		
		
		for user_histo in self.histo:
			try:
				user, init_state, init_r, u_history, r = self.sample_histo_v5(user_histo, nb_states, pivot_rating)
				users.append(user)
				initial_states.append(init_state)
				initial_rewards.append(init_r)
				user_histories.append(u_history)
				rewards.append(r)
				
			except:
				continue
				
		train_data = pd.DataFrame()
		test_data = pd.DataFrame()

		train_data['user'] = users
		train_data['state'] = initial_states
		train_data['state_reward'] = initial_rewards
		train_data_history = [u_h[0:int(train_test_ratio*len(u_h))] for u_h in user_histories]
		train_data['history'] = train_data_history
		train_data_rewards = [u_h[0:int(train_test_ratio*len(u_h))] for u_h in rewards]
		train_data['rewards'] = train_data_rewards

		test_data['user'] = users
		test_data_state = []
		for i in range(len(train_data_history)):
			row = train_data_history[i]
			if len(row) >= nb_states:
				test_data_state.append(row[-nb_states:])
			else:
				temp = initial_states[i][len(row)-nb_states:]
				temp.extend(row)
				test_data_state.append(temp)
		test_data['state'] =  test_data_state       
		test_data['history'] = [u_h[int(train_test_ratio*len(u_h)):] for u_h in user_histories]
		test_data_rewards = [u_h[int(train_test_ratio*len(u_h)):] for u_h in rewards]
		test_data['rewards'] = test_data_rewards
		
		print(len(train_data),len(test_data))
		
# 		train_data.to_csv('./data/ml-100k/train_session.csv', index=False)
# 		test_data.to_csv('./data/ml-100k/test_session.csv', index=False)
		if train_filename != None and test_filename != None: 
			train_data.to_csv(train_filename, index=False)
			test_data.to_csv(test_filename, index=False)



