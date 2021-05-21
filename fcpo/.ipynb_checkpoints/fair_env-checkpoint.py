import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
from data_util import *
import torch

class Environment ():
	def __init__(self, data, nb_user, nb_item, item_embeddings, user_embeddings, item_indicator, gamma, device, frac):
		self.data = data
		self.data_length = len(data)
		self.nb_user = nb_user
		self.nb_item = nb_item
		self.item_embeddings = item_embeddings
		self.user_embeddings = user_embeddings
		self.item_indicator = torch.from_numpy(item_indicator).float().to(device)
		self.gamma = gamma
		self.device = device
		self.frac = frac
		self.current_state = self.reset()


# 	def reset(self):
# 		self.current_user = self.data['user'].to_list()
# 		self.current_state = self.item_embeddings[self.data['state'].to_list()] 
# 		user_history = []
# 		for u_h in self.data['history'].values:
# 			h = torch.LongTensor(u_h)
# 			h_onehot = torch.FloatTensor(self.nb_item).zero_()
# 			h_onehot.scatter_(0, h, 1)
# 			user_history.append(h_onehot)
# 		self.current_user_history = torch.stack(user_history).to(self.device)
# 		return self.current_state

	def reset(self):
		current_data = self.data.sample(frac=self.frac)
		self.current_user = current_data['user'].to_list()
		self.nb_user = len(self.current_user)
		self.current_state = self.item_embeddings[current_data['state'].to_list()] 
		user_history = []
		for u_h in current_data['history'].values:
			h = torch.LongTensor(u_h)
			h_onehot = torch.FloatTensor(self.nb_item).zero_()
			h_onehot.scatter_(0, h, 1)
			user_history.append(h_onehot)
		self.current_user_history = torch.stack(user_history).to(self.device)
		return self.current_state

	def step(self, actions, item_idxes):
		costs = torch.sum(self.item_indicator[item_idxes],1)
		total_rewards = torch.zeros(self.nb_user).to(self.device)
		info = []
		
		for i in range(item_idxes.shape[1]):
			item_idxes_onehot = torch.FloatTensor(self.nb_user,self.nb_item).zero_().to(self.device)
			item_idxes_onehot.scatter_(1, item_idxes[:,i].view(-1,1), 1)

			results_onehot = torch.mul(self.current_user_history, item_idxes_onehot)
			info.append(torch.sum(results_onehot, 1))
			total_rewards += (self.gamma**i) * torch.sum(results_onehot, 1)

			mask =  torch.FloatTensor([0.5]*self.nb_user).view(-1,1).to(self.device)    
			masked_res = torch.cat((results_onehot, mask),dim=1)
			results = torch.argmax(masked_res,1)

			for j in range(len(results)):
				k = results[j]
				if k != self.nb_item:
					temp = np.append (self.current_state[j], [self.item_embeddings[k]], axis = 0)
					self.current_state[j] = np.delete (temp, 0, axis = 0)
					
			self.current_user_history = torch.mul(self.current_user_history, 1 - item_idxes_onehot)
			
		res = torch.stack(info).view(-1,item_idxes.shape[1])
		return self.current_state, total_rewards, costs, res